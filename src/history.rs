use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryItem {
    pub id: String,
    pub prompt: String,
    pub enhanced_prompt: Option<String>,
    pub aspect_ratio: String,
    pub width: u32,
    pub height: u32,
    pub image_url: String,
    pub created_at: String,
    pub duration_ms: u64,
    pub backend: String,
    pub seed: u64,
}

#[derive(Debug, Clone)]
pub struct HistoryStore {
    file_path: PathBuf,
    items: Arc<RwLock<Vec<HistoryItem>>>,
}

impl HistoryStore {
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        let file_path = path.as_ref().to_path_buf();
        let mut loaded = Vec::new();

        if let Ok(content) = fs::read_to_string(&file_path) {
            if let Ok(parsed) = serde_json::from_str::<Vec<HistoryItem>>(&content) {
                loaded = parsed;
            }
        }

        Self {
            file_path,
            items: Arc::new(RwLock::new(loaded)),
        }
    }

    pub async fn add(&self, item: HistoryItem) {
        let mut lock = self.items.write().await;
        // Insert newest at the beginning
        lock.insert(0, item);
        self.save_to_disk(&lock);
    }

    pub async fn get_all(&self) -> Vec<HistoryItem> {
        let lock = self.items.read().await;
        lock.clone()
    }

    pub async fn delete(&self, id: &str) -> bool {
        let mut lock = self.items.write().await;
        let original_len = lock.len();
        lock.retain(|item| item.id != id);
        let removed = lock.len() < original_len;
        if removed {
            self.save_to_disk(&lock);
        }
        removed
    }

    #[allow(dead_code)]
    pub async fn clear(&self) {
        let mut lock = self.items.write().await;
        lock.clear();
        self.save_to_disk(&lock);
    }

    fn save_to_disk(&self, items: &[HistoryItem]) {
        if let Some(parent) = self.file_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(items) {
            let _ = fs::write(&self.file_path, json);
        }
    }
}
