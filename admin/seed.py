"""Seed script: python -m admin.seed"""
import random
from datetime import datetime, timedelta, timezone
from admin.database import engine, SessionLocal, Base
from admin.models.role import Role
from admin.models.user import User
from admin.models.sector import Sector
from admin.models.order import Order
from admin.models.audit_log import AuditLog
from admin.models.refresh_token import RefreshToken
from admin.services.auth_service import hash_password

SECTORS = [
    ("auto_repair", "Auto Repair", "Vehicle maintenance and repair services", "wrench"),
    ("beauty_salon", "Beauty Salon", "Hair, nails, and beauty treatments", "scissors"),
    ("education_tutoring", "Education & Tutoring", "Learning and tutoring services", "graduation-cap"),
    ("event_planning", "Event Planning", "Event organization and coordination", "calendar"),
    ("financial_services", "Financial Services", "Banking, investment, and financial planning", "dollar-sign"),
    ("fitness_gym", "Fitness & Gym", "Workout, training, and fitness classes", "dumbbell"),
    ("food_delivery", "Food Delivery", "Restaurant meals delivered to your door", "utensils"),
    ("healthcare", "Healthcare", "Medical appointments and health services", "heart-pulse"),
    ("home_services", "Home Services", "Cleaning, plumbing, electrical work", "home"),
    ("insurance", "Insurance", "Insurance quotes and policy management", "shield"),
    ("it_services", "IT Services", "Tech support and IT solutions", "monitor"),
    ("laundry_services", "Laundry Services", "Wash, dry clean, and ironing", "shirt"),
    ("legal_services", "Legal Services", "Legal consultation and representation", "scale"),
    ("moving_services", "Moving Services", "Packing, moving, and storage", "truck"),
    ("pet_services", "Pet Services", "Grooming, boarding, and veterinary care", "paw-print"),
    ("photography", "Photography", "Photo and video sessions", "camera"),
    ("real_estate", "Real Estate", "Property listings and agent services", "building"),
    ("transportation", "Transportation", "Rides, deliveries, and logistics", "car"),
    ("travel_hotel", "Travel & Hotel", "Hotel bookings and travel arrangements", "plane"),
]

STATUSES = ["pending", "confirmed", "in_progress", "completed", "cancelled"]

FIRST_NAMES = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
               "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
               "Thomas", "Sarah", "Christopher", "Karen"]

LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
              "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
              "Thomas", "Taylor", "Moore", "Jackson", "Martin"]

ITEMS_BY_SECTOR = {
    "food_delivery": [
        [{"name": "Cheeseburger", "qty": 2, "price": 12.99}, {"name": "Fries", "qty": 1, "price": 4.99}],
        [{"name": "Margherita Pizza", "qty": 1, "price": 16.99}, {"name": "Garlic Bread", "qty": 1, "price": 5.49}],
        [{"name": "Pad Thai", "qty": 1, "price": 14.50}, {"name": "Spring Rolls", "qty": 2, "price": 6.00}],
    ],
    "healthcare": [
        [{"name": "General Checkup", "qty": 1, "price": 150.00}],
        [{"name": "Blood Test Panel", "qty": 1, "price": 85.00}, {"name": "Consultation", "qty": 1, "price": 120.00}],
        [{"name": "Physical Therapy Session", "qty": 3, "price": 90.00}],
    ],
    "beauty_salon": [
        [{"name": "Haircut & Style", "qty": 1, "price": 45.00}],
        [{"name": "Manicure", "qty": 1, "price": 30.00}, {"name": "Pedicure", "qty": 1, "price": 35.00}],
        [{"name": "Hair Coloring", "qty": 1, "price": 120.00}],
    ],
    "auto_repair": [
        [{"name": "Oil Change", "qty": 1, "price": 49.99}, {"name": "Tire Rotation", "qty": 1, "price": 29.99}],
        [{"name": "Brake Inspection", "qty": 1, "price": 89.00}],
        [{"name": "Engine Diagnostics", "qty": 1, "price": 120.00}],
    ],
    "fitness_gym": [
        [{"name": "Monthly Membership", "qty": 1, "price": 49.99}],
        [{"name": "Personal Training (5 sessions)", "qty": 1, "price": 250.00}],
        [{"name": "Yoga Class Pack (10)", "qty": 1, "price": 120.00}],
    ],
}

DEFAULT_ITEMS = [
    [{"name": "Standard Service", "qty": 1, "price": 75.00}],
    [{"name": "Premium Package", "qty": 1, "price": 150.00}],
    [{"name": "Consultation", "qty": 1, "price": 99.00}, {"name": "Follow-up", "qty": 1, "price": 50.00}],
]


def seed():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        # Clear tables
        db.query(AuditLog).delete()
        db.query(RefreshToken).delete()
        db.query(Order).delete()
        db.query(User).delete()
        db.query(Sector).delete()
        db.query(Role).delete()
        db.commit()

        # Roles
        roles = [
            Role(id=1, name="admin", display_name="Administrator", permissions={
                "users": ["create", "read", "update", "delete"],
                "orders": ["create", "read", "update", "delete"],
                "sectors": ["read", "update"],
                "export": ["csv", "pdf"],
                "dashboard": ["read"],
            }),
            Role(id=2, name="manager", display_name="Manager", permissions={
                "orders": ["read", "update", "delete"],
                "sectors": ["read"],
                "export": ["csv", "pdf"],
                "dashboard": ["read"],
            }),
            Role(id=3, name="viewer", display_name="Viewer", permissions={
                "orders": ["read"],
                "sectors": ["read"],
                "dashboard": ["read"],
            }),
        ]
        db.add_all(roles)
        db.commit()
        print(f"Seeded {len(roles)} roles")

        # Users
        hashed_pw = hash_password("Admin123!")
        users = [
            User(id=1, email="admin@admin.com", password_hash=hashed_pw, name="Admin User",
                 phone="+1-555-0001", role_id=1, is_active=True, is_email_verified=True),
        ]
        for i in range(2, 7):
            users.append(User(
                id=i, email=f"manager{i-1}@example.com", password_hash=hash_password("Manager1!"),
                name=f"{FIRST_NAMES[i]} {LAST_NAMES[i]}", phone=f"+1-555-{1000+i:04d}",
                role_id=2, is_active=True, is_email_verified=True,
            ))
        for i in range(7, 21):
            users.append(User(
                id=i, email=f"viewer{i-6}@example.com", password_hash=hash_password("Viewer123!"),
                name=f"{FIRST_NAMES[i % len(FIRST_NAMES)]} {LAST_NAMES[i % len(LAST_NAMES)]}",
                phone=f"+1-555-{1000+i:04d}",
                role_id=3, is_active=True, is_email_verified=True,
            ))
        db.add_all(users)
        db.commit()
        print(f"Seeded {len(users)} users")

        # Sectors
        sector_objs = []
        for idx, (name, display, desc, icon) in enumerate(SECTORS, 1):
            sector_objs.append(Sector(id=idx, name=name, display_name=display, description=desc, icon=icon, is_active=True))
        db.add_all(sector_objs)
        db.commit()
        print(f"Seeded {len(sector_objs)} sectors")

        # Orders
        orders = []
        now = datetime.now(timezone.utc)
        order_id = 1
        for sector_name, _, _, _ in SECTORS:
            count = random.randint(6, 10)
            for _ in range(count):
                items_pool = ITEMS_BY_SECTOR.get(sector_name, DEFAULT_ITEMS)
                items = random.choice(items_pool)
                total = sum(item["price"] * item["qty"] for item in items)
                days_ago = random.randint(0, 90)
                created = now - timedelta(days=days_ago, hours=random.randint(0, 23), minutes=random.randint(0, 59))
                fname = random.choice(FIRST_NAMES)
                lname = random.choice(LAST_NAMES)
                orders.append(Order(
                    id=order_id,
                    customer_name=f"{fname} {lname}",
                    customer_phone=f"+1-555-{random.randint(1000,9999)}",
                    customer_email=f"{fname.lower()}.{lname.lower()}@email.com",
                    sector=sector_name,
                    status=random.choice(STATUSES),
                    items=items,
                    total_amount=round(total, 2),
                    notes=random.choice([None, "Rush order", "Call before delivery", "Special instructions", "Gift wrap requested", ""]),
                    metadata_={"source": random.choice(["web", "phone", "app"]), "channel": random.choice(["voice", "sms", "chat"])},
                ))
                order_id += 1
        db.add_all(orders)
        db.commit()
        print(f"Seeded {len(orders)} orders")

        # Audit logs
        audit_actions = [
            ("create", "order"), ("update", "order"), ("delete", "order"),
            ("create", "user"), ("update", "user"),
            ("bulk_update", "order"), ("bulk_delete", "order"),
            ("login", "user"), ("export", "order"),
        ]
        logs = []
        for i in range(1, 21):
            action, entity = random.choice(audit_actions)
            logs.append(AuditLog(
                user_id=random.choice([1, 2, 3]),
                action=action,
                entity_type=entity,
                entity_id=random.randint(1, 50),
                details={"ip": f"192.168.1.{random.randint(1,254)}", "action_detail": f"Performed {action} on {entity}"},
            ))
        db.add_all(logs)
        db.commit()
        print(f"Seeded {len(logs)} audit logs")

        print("\nSeed complete! Default admin login: admin@admin.com / Admin123!")

    finally:
        db.close()


if __name__ == "__main__":
    seed()
