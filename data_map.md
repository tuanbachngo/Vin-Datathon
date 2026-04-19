# Data Map

| table_name | grain | primary_key |
|---|---|---|
| customers | One row per customer | customer_id |
| geography | One row per zip | zip |
| inventory | One row per product snapshot period | (none) |
| inventory_enhanced | Optional enhanced inventory table | (none) |
| order_items | One row per order line item | (none) |
| orders | One row per order | order_id |
| payments | One row per order payment | order_id |
| products | One row per product | product_id |
| promotions | One row per promotion | promo_id |
| returns | One row per return event | return_id |
| reviews | One row per review | review_id |
| sales | One row per date | Date |
| sample_submission | Submission template | (none) |
| shipments | One row per shipment | (none) |
| web_traffic | One row per date and traffic source | (none) |