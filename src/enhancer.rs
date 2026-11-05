//! Prompt enhancement module for FLUX.2 Klein.
//! Inspired by the chirpr.in context: rewriting terse prompts (e.g., "lakshmi good morning")
//! into rich, culturally accurate descriptors with lighting, style, and composition details.

pub fn enhance_prompt(raw: &str) -> String {
    let lower = raw.trim().to_lowercase();

    if lower.contains("lakshmi") || (lower.contains("good morning") && (lower.contains("god") || lower.contains("goddess"))) {
        return format!(
            "Goddess Lakshmi with four arms, seated gracefully upon an open pink lotus, \
            wearing rich crimson red silk saree with ornate golden zari borders, \
            intricate temple gold jewellery, shower of glowing gold coins cascading from right palm, \
            flanked by two auspicious white royal elephants offering gentle water ablution, \
            soft warm morning sunrise rays piercing through mystical mist, \
            traditional Indian calendar art style inspired by Raja Ravi Varma, \
            ornate auspicious devotional glow, high resolution, 8k masterpiece. (Original: {})",
            raw
        );
    }

    if lower.contains("tirupati") || lower.contains("balaji") || lower.contains("venkateswara") {
        return format!(
            "Lord Sri Venkateswara Swamy of Tirumala Tirupati in divine darshan alankaram, \
            adorned with sparkling diamond-studded golden Kireetam and protective golden kavacham, \
            Shanku and Chakra glowing on shoulders, pure white sacred Namam with Kasturi tilakam on forehead, \
            cascading fragrant garlands of sacred tulasi leaves and fresh yellow marigolds, \
            sanctum sanctorum illuminated by traditional flickering brass oil deepams, \
            profound spiritual aura, photorealistic sacred temple darshan, 8k resolution. (Original: {})",
            raw
        );
    }

    if lower.contains("ganesh") || lower.contains("vinayak") || lower.contains("ganapati") {
        return format!(
            "Lord Ganesha with gentle benevolent eyes and ornate golden trunk, \
            seated comfortably upon a carved wooden peetam, holding a golden modak and lotus blossom, \
            wearing saffron dhoti and radiant jeweled crown, surrounded by fresh red hibiscus flowers, \
            burning fragrant agarbatti smoke curls and warm diya lamps, \
            vibrant Indian festive calendar art, divine blessings, highly detailed. (Original: {})",
            raw
        );
    }

    if lower.contains("shiva") || lower.contains("jyotirlinga") || lower.contains("mahadev") {
        return format!(
            "Sacred holy Jyotirlinga enveloped in holy tripundra bhasma, fresh bilva patra leaves and white datura flowers, \
            radiant glowing golden snake coiled gently around the lingam, \
            mystic Himalayan mountain backdrop at serene twilight, cosmic trishul and damru standing nearby, \
            ethereal blue and golden meditative lighting, sacred spiritual atmosphere. (Original: {})",
            raw
        );
    }

    if lower.contains("neon") || lower.contains("24 hours") || lower.contains("alley") {
        return format!(
            "A luminous neon sign reading 'OPEN 24 HOURS' glowing in electric cyan and vibrant magenta \
            in a moody, rain-soaked city alley at midnight, hyper-detailed glistening reflections on wet asphalt pavement, \
            diffuse mist, steam rising from underground grates, cinematic cyberpunk atmosphere, shallow depth of field, 8k. (Original: {})",
            raw
        );
    }

    if lower.contains("cricket") || lower.contains("stadium") {
        return format!(
            "Electrifying cricket match moment under blinding stadium floodlights during a packed championship night, \
            batsman executing an exquisite drive, turf grass particles flying in ultra slow-motion, \
            crowd in stands roaring with team flags, dramatic sports photojournalism, cinematic lighting, 8k. (Original: {})",
            raw
        );
    }

    if lower.contains("good morning") {
        return format!(
            "Serene Indian countryside morning greeting scene, golden sunrise light filtering through ancient banyan and mango trees, \
            dewdrops sparkling on fresh green leaves, traditional brass filter coffee tumbler steaming gently on a verandah, \
            warm cheerful peaceful morning ambience, soft bokeh, cinematic photograph. (Original: {})",
            raw
        );
    }

    // Default general enhancer
    format!(
        "{}, beautifully composed with cinematic lighting, fine details, harmonious color palette, high fidelity, 8k resolution",
        raw.trim()
    )
}

