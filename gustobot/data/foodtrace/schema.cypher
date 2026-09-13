CREATE CONSTRAINT foodtrace_ingredient_lot_id IF NOT EXISTS
FOR (node:FoodTraceIngredientLot)
REQUIRE node.ingredient_lot_id IS UNIQUE;

CREATE CONSTRAINT foodtrace_recipe_id IF NOT EXISTS
FOR (node:FoodTraceRecipe)
REQUIRE node.recipe_id IS UNIQUE;

CREATE CONSTRAINT foodtrace_product_id IF NOT EXISTS
FOR (node:FoodTraceProduct)
REQUIRE node.product_id IS UNIQUE;

CREATE CONSTRAINT foodtrace_batch_id IF NOT EXISTS
FOR (node:FoodTraceProductionBatch)
REQUIRE node.production_batch_id IS UNIQUE;
