CREATE TABLE IF NOT EXISTS foodtrace_store_inventory (
    inventory_id VARCHAR(64) PRIMARY KEY,
    store_id VARCHAR(64) NOT NULL,
    production_batch_id VARCHAR(64) NOT NULL,
    quantity INT NOT NULL,
    INDEX idx_foodtrace_inventory_batch (production_batch_id)
);

CREATE TABLE IF NOT EXISTS foodtrace_order_exposures (
    order_id VARCHAR(64) PRIMARY KEY,
    store_id VARCHAR(64) NOT NULL,
    production_batch_id VARCHAR(64) NOT NULL,
    quantity INT NOT NULL,
    customer_ref VARCHAR(128) NOT NULL,
    ordered_at DATETIME NOT NULL,
    INDEX idx_foodtrace_order_batch (production_batch_id)
);

DELETE FROM foodtrace_store_inventory
WHERE inventory_id IN (
    'INV-FT-001', 'INV-FT-002', 'INV-FT-003', 'INV-FT-004', 'INV-FT-005'
);

INSERT INTO foodtrace_store_inventory (
    inventory_id, store_id, production_batch_id, quantity
) VALUES
    ('INV-FT-001', 'STORE-FT-001', 'BATCH-FT-001', 13),
    ('INV-FT-002', 'STORE-FT-001', 'BATCH-FT-004', 30),
    ('INV-FT-003', 'STORE-FT-002', 'BATCH-FT-002', 24),
    ('INV-FT-004', 'STORE-FT-003', 'BATCH-FT-003', 18),
    ('INV-FT-005', 'STORE-FT-003', 'BATCH-FT-004', 12);

DELETE FROM foodtrace_order_exposures
WHERE order_id IN (
    'ORDER-FT-001', 'ORDER-FT-002', 'ORDER-FT-003', 'ORDER-FT-004'
);

INSERT INTO foodtrace_order_exposures (
    order_id, store_id, production_batch_id, quantity, customer_ref, ordered_at
) VALUES
    (
        'ORDER-FT-001', 'STORE-FT-001', 'BATCH-FT-001', 1,
        'CUSTOMER-HASH-001', '2026-07-23 12:00:00'
    ),
    (
        'ORDER-FT-002', 'STORE-FT-002', 'BATCH-FT-002', 2,
        'CUSTOMER-HASH-002', '2026-07-24 12:00:00'
    ),
    (
        'ORDER-FT-003', 'STORE-FT-003', 'BATCH-FT-003', 1,
        'CUSTOMER-HASH-003', '2026-07-25 12:00:00'
    ),
    (
        'ORDER-FT-004', 'STORE-FT-002', 'BATCH-FT-004', 1,
        'CUSTOMER-HASH-004', '2026-07-25 13:00:00'
    );
