# Olist E-Commerce Dataset Description

This document describes the 9 datasets used in the E-Commerce 360 Dashboard. These files are located in the `data/` directory.

## 1. Customers Dataset (`olist_customers_dataset.csv`)
**Purpose**: Contains information about the customer and their location.
- **Key Columns**:
    - `customer_id`: Key to the orders dataset. Each order has a unique customer_id.
    - `customer_unique_id`: Unique identifier of a customer.
    - `customer_zip_code_prefix`: First 5 digits of the zip code.
    - `customer_city`: Customer city name.
    - `customer_state`: Customer state code.

## 2. Geolocation Dataset (`olist_geolocation_dataset.csv`)
**Purpose**: Maps zip codes to lat/lng coordinates for geospatial analysis.
- **Key Columns**:
    - `geolocation_zip_code_prefix`: First 5 digits of zip code.
    - `geolocation_lat`: Latitude.
    - `geolocation_lng`: Longitude.
    - `geolocation_city`: City name.
    - `geolocation_state`: State code.

## 3. Order Items Dataset (`olist_order_items_dataset.csv`)
**Purpose**: Details items within each order. An order can have multiple items (relationship is 1:N).
- **Key Columns**:
    - `order_id`: Order unique identifier.
    - `order_item_id`: Sequential number identifying number of items in the same order.
    - `product_id`: Product unique identifier.
    - `seller_id`: Seller unique identifier.
    - `price`: Item price.
    - `freight_value`: Item freight value.

## 4. Order Payments Dataset (`olist_order_payments_dataset.csv`)
**Purpose**: Payment methods and installment details.
- **Key Columns**:
    - `order_id`: Unique identifier of an order.
    - `payment_sequential`: Sequence of the payment.
    - `payment_type`: Method (credit_card, boleto, voucher, etc.).
    - `payment_installments`: Number of installments chosen.
    - `payment_value`: Transaction value.

## 5. Order Reviews Dataset (`olist_order_reviews_dataset.csv`)
**Purpose**: Customer satisfaction data.
- **Key Columns**:
    - `review_id`: Unique review identifier.
    - `order_id`: Unique order identifier.
    - `review_score`: 1 to 5 rating.
    - `review_comment_title`: Title of the review.
    - `review_comment_message`: Text content of the review.
    - `review_creation_date`: Date the review was sent.

## 6. Orders Dataset (`olist_orders_dataset.csv`)
**Purpose**: The core dataset connecting all others. Tracks order status and timestamps.
- **Key Columns**:
    - `order_id`: Unique identifier of the order.
    - `customer_id`: Key to the customer dataset.
    - `order_status`: Status (delivered, shipped, etc.).
    - `order_purchase_timestamp`: Purchase timestamp.
    - `order_approved_at`: Payment approval timestamp.
    - `order_delivered_carrier_date`: Order posting timestamp.
    - `order_delivered_customer_date`: Actual delivery date.
    - `order_estimated_delivery_date`: Estimated delivery date.

## 7. Products Dataset (`olist_products_dataset.csv`)
**Purpose**: Product metadata.
- **Key Columns**:
    - `product_id`: Unique product identifier.
    - `product_category_name`: Root category name (in Portuguese).
    - `product_weight_g`: Product weight in grams.
    - `product_length_cm`: Product length in cm.
    - `product_height_cm`: Product height in cm.
    - `product_width_cm`: Product width in cm.

## 8. Sellers Dataset (`olist_sellers_dataset.csv`)
**Purpose**: Seller location and identity.
- **Key Columns**:
    - `seller_id`: Seller unique identifier.
    - `seller_zip_code_prefix`: Zip code prefix.
    - `seller_city`: Seller city name.
    - `seller_state`: Seller state code.

## 9. Category Translation Dataset (`product_category_name_translation.csv`)
**Purpose**: Translates category names to English.
- **Key Columns**:
    - `product_category_name`: Category name in Portuguese.
    - `product_category_name_english`: Category name in English.
