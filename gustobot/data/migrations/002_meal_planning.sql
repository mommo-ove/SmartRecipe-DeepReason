CREATE TABLE IF NOT EXISTS meal_planning_recipe_facts (
    recipe_id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    meal_types JSON NOT NULL,
    calories_kcal DECIMAL(10,2) NOT NULL,
    protein_g DECIMAL(10,2) NOT NULL,
    total_minutes INT NOT NULL,
    estimated_cost_cents INT NOT NULL,
    allergens JSON NOT NULL,
    allergen_status_verified BOOLEAN NOT NULL DEFAULT FALSE,
    preference_tags JSON NOT NULL,
    source_refs JSON NOT NULL,
    planning_eligible BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_meal_planning_eligible (planning_eligible)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS meal_planning_ingredient_amounts (
    recipe_id VARCHAR(64) NOT NULL,
    ingredient_id VARCHAR(128) NOT NULL,
    amount DECIMAL(12,3) NOT NULL,
    unit ENUM('g', 'ml') NOT NULL,
    ingredient_order INT NOT NULL DEFAULT 0,
    PRIMARY KEY (recipe_id, ingredient_id),
    INDEX idx_meal_ingredient_recipe_order (recipe_id, ingredient_order),
    CONSTRAINT fk_meal_ingredient_recipe
        FOREIGN KEY (recipe_id) REFERENCES meal_planning_recipe_facts(recipe_id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
