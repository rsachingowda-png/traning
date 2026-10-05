import json
import random

random.seed(42)

NAMES = ["Sachin", "Sac", "Niru", "Sush", "Amit", "Priya", "Rahul", "Kavya"]
PRODUCTS = ["Wireless Mouse", "Mechanical Keyboard", "USB-C Hub", "4K Monitor", "Ergonomic Chair", "Noise Canceling Headphones"]
STATUSES = ["Delivered", "In Transit", "Processing", "Cancelled", "Returned"]
CITIES = ["Bengaluru", "Mumbai", "Delhi", "Hyderabad", "Pune", "Mysuru"]

def generate_sample(order_id):
    name = random.choice(NAMES)
    product = random.choice(PRODUCTS)
    status = random.choice(STATUSES)
    city = random.choice(CITIES)
    amount = random.randint(1500, 45000)
    
    # Input statement
    context_input = f"Customer {name} (Order ID: ORD-{order_id}) purchased a {product} for Rs. {amount}. Current shipping destination is {city} and status is marked as {status}."
    
    instruction = "Extract order details into exact JSON format including refund eligibility (eligible only if status is Delivered or Returned)."
    
    refund_eligible = status in ["Delivered", "Returned"]
    
    expected_output = json.dumps({
        "order_id": f"ORD-{order_id}",
        "customer_name": name,
        "item": product,
        "amount_inr": amount,
        "destination": city,
        "status": status,
        "refund_eligible": refund_eligible
    }, indent=2)

    return {
        "instruction": instruction,
        "input": context_input,
        "output": expected_output
    }

def main():
    # 80 training samples, 20 test samples
    train_data = [generate_sample(1000 + i) for i in range(80)]
    test_data = [generate_sample(2000 + i) for i in range(20)]
    
    with open("train.jsonl", "w") as f:
        for entry in train_data:
            f.write(json.dumps(entry) + "\n")
            
    with open("test.jsonl", "w") as f:
        for entry in test_data:
            f.write(json.dumps(entry) + "\n")
            
    print("Generated train.jsonl (80 samples) and test.jsonl (20 samples).")

if __name__ == "__main__":
    main()
