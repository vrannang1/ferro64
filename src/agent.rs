use crate::db::{AgentStep, AgentTask, Database};
use crate::llm::{ChatMessagePayload, ChatOptions, Effort, LlmEngine, ModelSpec, REASONING_EXHAUSTED};
use serde::Deserialize;
use std::sync::Arc;
use std::time::Instant;
use uuid::Uuid;

#[derive(Deserialize)]
pub struct RunAgentTaskRequest {
    pub goal: String,
    pub planner_model: Option<String>,
    pub reviewer_model: Option<String>,
    pub synthesizer_model: Option<String>,
    #[serde(default)]
    pub effort: Effort,
    #[serde(default)]
    pub thinking: bool,
}

#[derive(Clone)]
pub struct AgentHarness {
    llm: Arc<LlmEngine>,
    db: Database,
}

impl AgentHarness {
    pub fn new(llm: Arc<LlmEngine>, db: Database) -> Self {
        Self { llm, db }
    }

    /// This is a bounded draft/review/synthesis workflow. Models exchange text;
    /// they do not execute tools or gain filesystem/network access.
    pub async fn execute_task(&self, req: RunAgentTaskRequest) -> Result<AgentTask, String> {
        let harness = self.clone();
        // Complete and persist the run even if its HTTP client disconnects.
        tokio::spawn(async move { harness.execute_task_inner(req).await })
            .await
            .map_err(|error| format!("Agent worker failed: {error}"))?
    }

    async fn execute_task_inner(&self, req: RunAgentTaskRequest) -> Result<AgentTask, String> {
        let goal = req.goal.trim().to_string();
        if goal.is_empty() {
            return Err("Enter an agent objective".to_string());
        }
        if goal.chars().count() > 50_000 {
            return Err("Objective must be at most 50,000 characters".to_string());
        }
        let start = Instant::now();
        let mut task = AgentTask {
            id: Uuid::new_v4().to_string(),
            goal,
            status: "running".to_string(),
            steps: Vec::new(),
            final_result: None,
            created_at: chrono::Utc::now().to_rfc3339(),
            completed_at: None,
            total_duration_ms: 0,
        };
        self.db
            .save_agent_task(&task)
            .map_err(|e| format!("Cannot save agent task: {e}"))?;
        let outcome = self.run_stages(&req, &mut task, start).await;
        task.total_duration_ms = start.elapsed().as_millis() as u64;
        task.completed_at = Some(chrono::Utc::now().to_rfc3339());
        match outcome {
            Ok(()) => {
                task.status = "completed".to_string();
                self.db
                    .save_agent_task(&task)
                    .map_err(|e| format!("Cannot save completed agent task {}: {e}", task.id))?;
                Ok(task)
            }
            Err(error) => {
                task.status = "failed".to_string();
                task.final_result = Some(format!("Task failed: {error}"));
                self.db
                    .save_agent_task(&task)
                    .map_err(|e| format!("{error}; could not persist failure: {e}"))?;
                Err(format!(
                    "Task {} failed: {error}. Completed steps are saved in run history.",
                    task.id
                ))
            }
        }
    }

    async fn run_stages(
        &self,
        req: &RunAgentTaskRequest,
        task: &mut AgentTask,
        start: Instant,
    ) -> Result<(), String> {
        let models = self.llm.models().await?;
        // Defaults chosen by testing: Qwen3.5 drafts and writes the result, and
        // Nemotron (another maker) reviews; it caught errors without adding new ones.
        // On smaller machines: Qwen3.5 9B does all three.
        let planner = choose_model(&models, req.planner_model.as_deref(), &["qwen3.5:35b-a3b", "qwen3.5:9b", "qwen3:8b"])?;
        let reviewer = choose_model(
            &models,
            req.reviewer_model.as_deref(),
            &["nemotron-3-nano:30b-a3b", "glm-4.7-flash", "qwen3.5:9b"],
        )?;
        let synthesizer = choose_model(&models, req.synthesizer_model.as_deref(), &["qwen3.5:35b-a3b", "qwen3.5:9b", "qwen3:8b"])?;

        let stages = [
            ("Planner", planner, "Develop a useful first draft of the requested deliverable. Identify requirements, constraints, and important assumptions. Produce substantive work toward the objective."),
            ("Reviewer", reviewer, "Review the draft against the original objective. Identify unsupported claims, errors, omissions, and edge cases. Give concrete corrections. Do not claim external verification or tool execution."),
            ("Synthesizer", synthesizer, "Produce the final deliverable for the objective. Incorporate valid review feedback and resolve issues in the draft. State remaining uncertainty. Return a complete, clear answer."),
        ];
        let options = ChatOptions {
            thinking: req.thinking,
            effort: req.effort,
            ..Default::default()
        };
        let instructions = crate::standing_instructions(&self.db)
            .map(|text| format!("\n\n{text}"))
            .unwrap_or_default();
        let mut outputs = Vec::new();
        for (index, (role, model, instruction)) in stages.into_iter().enumerate() {
            let mut prompt = format!("OBJECTIVE:\n{}", task.goal);
            if let Some(draft) = outputs.first() {
                prompt.push_str(&format!("\n\nINITIAL DRAFT:\n{draft}"));
            }
            if let Some(review) = outputs.get(1) {
                prompt.push_str(&format!("\n\nREVIEW:\n{review}"));
            }
            let messages = [
                ChatMessagePayload {
                    role: "system".to_string(),
                    content: format!("{instruction}\n{}{instructions}", req.effort.guidance()),
                    images: None,
                },
                ChatMessagePayload {
                    role: "user".to_string(),
                    content: prompt,
                    images: None,
                },
            ];
            // chat_completion owns the GPU queue through its final unload. A
            // failed unload aborts the workflow before another model is loaded.
            let mut attempt = self.llm.chat_completion(&model, &messages, options).await;
            // A model that reasons until the budget runs out gets one more
            // try without thinking and with the largest budget, rather than
            // failing the whole run.
            if matches!(&attempt, Err(error) if error.contains(REASONING_EXHAUSTED)) {
                let retry = ChatOptions { thinking: false, effort: Effort::High, ..options };
                attempt = self.llm.chat_completion(&model, &messages, retry).await;
            }
            let (output, duration_ms) = attempt.map_err(|error| format!("{role} ({model}): {error}"))?;
            task.steps.push(AgentStep {
                step_index: index + 1,
                role: role.to_string(),
                model,
                description: instruction.to_string(),
                output: output.clone(),
                duration_ms,
            });
            outputs.push(output);
            task.total_duration_ms = start.elapsed().as_millis() as u64;
            self.db
                .save_agent_task(task)
                .map_err(|e| format!("Cannot save {role} progress: {e}"))?;
        }
        task.final_result = outputs.pop();
        Ok(())
    }
}

fn choose_model(
    models: &[ModelSpec],
    requested: Option<&str>,
    preferred: &[&str],
) -> Result<String, String> {
    let eligible = |model: &&ModelSpec| {
        model.installed
            && model.capabilities.iter().any(|c| c == "chat")
            && !model.is_image_generator
            && !model.capabilities.iter().any(|c| c == "embedding")
    };
    if let Some(requested) = requested {
        return models.iter().filter(eligible).find(|model| model.id == requested)
            .map(|model| model.id.clone())
            .ok_or_else(|| format!("'{requested}' is not an installed chat model. Refresh the model inventory and choose an available text model."));
    }
    preferred
        .iter()
        .find_map(|id| models.iter().filter(eligible).find(|model| model.id == *id))
        .or_else(|| models.iter().filter(eligible).next())
        .map(|model| model.id.clone())
        .ok_or_else(|| {
            "No installed chat model is available. Download one in Models, then refresh."
                .to_string()
        })
}
