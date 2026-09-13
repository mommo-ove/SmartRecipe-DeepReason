MERGE (peanut:FoodTraceIngredientLot {ingredient_lot_id: 'LOT-PEANUT-001'})
SET peanut.ingredient_name = 'peanut paste',
    peanut.supplier_id = 'SUP-PEANUT-001',
    peanut.suspected_allergens = ['peanut'];

MERGE (chicken:FoodTraceIngredientLot {ingredient_lot_id: 'LOT-CHICKEN-001'})
SET chicken.ingredient_name = 'chicken',
    chicken.supplier_id = 'SUP-PROTEIN-001';

MERGE (noodle:FoodTraceIngredientLot {ingredient_lot_id: 'LOT-NOODLE-001'})
SET noodle.ingredient_name = 'rice noodle',
    noodle.supplier_id = 'SUP-GRAIN-001';

MERGE (tomato:FoodTraceIngredientLot {ingredient_lot_id: 'LOT-TOMATO-001'})
SET tomato.ingredient_name = 'tomato',
    tomato.supplier_id = 'SUP-PRODUCE-001';

MERGE (recipe1:FoodTraceRecipe {recipe_id: 'RECIPE-FT-001'})
SET recipe1.recipe_name = 'spicy chicken bowl';
MERGE (recipe2:FoodTraceRecipe {recipe_id: 'RECIPE-FT-002'})
SET recipe2.recipe_name = 'satay rice noodles';
MERGE (recipe3:FoodTraceRecipe {recipe_id: 'RECIPE-FT-003'})
SET recipe3.recipe_name = 'tomato soup';

MERGE (product1:FoodTraceProduct {product_id: 'PRODUCT-FT-001'})
SET product1.product_name = 'chicken bowl retail pack';
MERGE (product2:FoodTraceProduct {product_id: 'PRODUCT-FT-002'})
SET product2.product_name = 'satay noodle retail pack';
MERGE (product3:FoodTraceProduct {product_id: 'PRODUCT-FT-003'})
SET product3.product_name = 'tomato soup retail pack';

MERGE (batch1:FoodTraceProductionBatch {production_batch_id: 'BATCH-FT-001'});
MERGE (batch2:FoodTraceProductionBatch {production_batch_id: 'BATCH-FT-002'});
MERGE (batch3:FoodTraceProductionBatch {production_batch_id: 'BATCH-FT-003'});
MERGE (batch4:FoodTraceProductionBatch {production_batch_id: 'BATCH-FT-004'});

MERGE (peanut)-[:USED_IN_RECIPE]->(recipe1);
MERGE (chicken)-[:USED_IN_RECIPE]->(recipe1);
MERGE (peanut)-[:USED_IN_RECIPE]->(recipe2);
MERGE (noodle)-[:USED_IN_RECIPE]->(recipe2);
MERGE (tomato)-[:USED_IN_RECIPE]->(recipe3);

MERGE (recipe1)-[:PRODUCES]->(product1);
MERGE (recipe2)-[:PRODUCES]->(product2);
MERGE (recipe3)-[:PRODUCES]->(product3);

MERGE (product1)-[:HAS_BATCH]->(batch1);
MERGE (product1)-[:HAS_BATCH]->(batch2);
MERGE (product2)-[:HAS_BATCH]->(batch3);
MERGE (product3)-[:HAS_BATCH]->(batch4);
