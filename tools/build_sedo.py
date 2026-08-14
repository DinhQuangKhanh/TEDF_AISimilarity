"""Generate ``app/ontology/sedo.json`` from a compact, human-editable spec.

SEDO (paper §V-D) is a four-layer SE ontology: Technical Stack · Methodology · Domain
Entity · Task Type. Only the ``isA`` (parent) relation is used by the similarity measures
in ``app/ontology/sedo.py`` (Wu-Palmer, wpath). Concepts carry surface ``keywords`` (English
only — duplicate checking runs on English content) that Module-1 NER matches against a title.

The paper says SEDO "grows iteratively as the evaluation phase surfaces missing concepts",
so keeping the ontology as a spec + generator (rather than hand-edited JSON) makes that growth
cheap and keeps the file consistent. Edit ``SPEC`` below, then run:

    python tools/build_sedo.py

Spec format — a nested tree of ``name: (keywords, children)``:
    "React": (["react", "reactjs", "react.js"], {})           # leaf
    "Frontend": (["frontend", "front-end"], { ...children... }) # parent
"""

from __future__ import annotations

import json
import os
import re

# ── Layer 1: Technical Stack ────────────────────────────────────────────────────────
TECH = {
    "Frontend": (["frontend", "front-end", "client side"], {
        "React": (["react", "reactjs", "react.js", "reactts", "react system"], {}),
        "Angular": (["angular", "angularjs"], {}),
        "Vue": (["vue", "vuejs", "vue.js"], {}),
        "Svelte": (["svelte", "sveltekit"], {}),
        "Blazor": (["blazor", "blazor webassembly"], {}),
        "Next.js": (["next.js", "nextjs", "next js"], {}),
        "Razor Pages": (["razor pages", "razor"], {}),
        "Tailwind CSS": (["tailwind", "tailwindcss"], {}),
        "Vite": (["vite"], {}),
    }),
    "Backend": (["backend", "back-end", "server side"], {
        "Node.js": (["node.js", "nodejs", "node js", "express", "express.js", "expressjs"], {}),
        "NestJS": (["nestjs", "nest.js", "nest js"], {}),
        "Spring Boot": (["spring boot", "spring", "java spring", "spring mvc"], {}),
        "ASP.NET Core": (["asp.net core", "asp.net", "dotnet core", ".net core", ".net 8", "web api", "aspnet"], {}),
        "Django": (["django", "django rest"], {}),
        "Flask": (["flask"], {}),
        "FastAPI": (["fastapi", "fast api"], {}),
        "Laravel": (["laravel", "php laravel", "php"], {}),
    }),
    "Database": (["database"], {
        "PostgreSQL": (["postgresql", "postgres", "psql", "pgvector"], {}),
        "MySQL": (["mysql", "mariadb"], {}),
        "MongoDB": (["mongodb", "mongo", "nosql", "mongoose", "mongodb atlas"], {}),
        "SQL Server": (["sql server", "mssql", "sqlserver"], {}),
        "Redis": (["redis", "redis cache"], {}),
        "Firebase": (["firebase", "firestore", "firebase database"], {}),
        "Supabase": (["supabase"], {}),
    }),
    "Mobile": (["mobile", "mobile app"], {
        "Flutter": (["flutter", "dart flutter", "dart"], {}),
        "React Native": (["react native", "reactnative", "expo"], {}),
        "Android": (["android", "kotlin"], {}),
        "iOS": (["ios", "swift", "iphone app"], {}),
    }),
    "DevOps": (["devops", "ci/cd", "cicd"], {
        "Docker": (["docker", "container", "dockerize"], {}),
        "Kubernetes": (["kubernetes", "k8s"], {}),
        "CI/CD Pipeline": (["jenkins", "github actions", "gitlab ci", "pipeline"], {}),
        "Cloud": (["aws", "amazon web services", "azure", "vercel", "render", "cloud", "nginx", "vps"], {}),
    }),
    "Integration": (["integration"], {
        "REST API": (["rest api", "rest", "restful", "web service"], {}),
        "GraphQL": (["graphql", "graph ql"], {}),
        "gRPC": (["grpc"], {}),
        "Realtime": (["websocket", "signalr", "socket.io", "socketio", "realtime"], {}),
        "Message Queue": (["kafka", "rabbitmq", "message queue", "message broker"], {}),
    }),
    "AI/ML": (["ai", "artificial intelligence", "machine learning"], {
        "Machine Learning": (["machine learning", "scikit-learn", "sklearn", "xgboost", "prediction model", "forecasting model"], {}),
        "Deep Learning": (["deep learning", "neural network", "pytorch", "tensorflow"], {}),
        "NLP": (["nlp", "natural language processing", "sentiment analysis", "text analysis"], {}),
        "Computer Vision": (["computer vision", "image recognition", "object detection"], {}),
        "LLM/RAG": (["llm", "rag", "langchain", "openai", "gemini", "chatbot", "generative ai", "large language model"], {}),
        "OCR": (["ocr", "tesseract", "optical character recognition"], {}),
    }),
}

# ── Layer 2: Domain Entity (business domains) ───────────────────────────────────────
DOMAIN = {
    "Hospitality": (["hospitality"], {
        "Hotel": (["hotel", "lodging"], {}),
        "Restaurant": (["restaurant", "dining", "canteen", "food ordering"], {}),
        "Homestay": (["homestay", "guest house"], {}),
        "Tourism": (["tourism", "travel", "tour", "itinerary", "waterway tourism", "eco-tourism"], {}),
        "Coworking Space": (["coworking", "co-working", "workspace booking"], {}),
    }),
    "Healthcare": (["healthcare", "medical"], {
        "Hospital": (["hospital"], {}),
        "Clinic": (["clinic", "outpatient", "medical appointment"], {}),
        "Pharmacy": (["pharmacy", "drugstore", "medicine", "medication"], {}),
        "Laboratory": (["laboratory", "lab", "diagnostics"], {}),
        "Mental Health": (["mental health", "psychology"], {}),
        "Veterinary": (["veterinary", "pet"], {}),
        "Elderly Care": (["elderly", "elderly care"], {}),
        "Disease Diagnosis": (["diagnosis", "disease detection", "pulmonary", "lung disease"], {}),
    }),
    "Education": (["education", "learning", "e-learning"], {
        "School": (["school", "k12", "preschool", "kindergarten"], {}),
        "University": (["university", "campus", "higher education", "fpt university"], {}),
        "Course": (["course", "e-learning", "online learning", "lms"], {}),
        "Library": (["library", "book borrowing"], {}),
        "Tutoring": (["tutoring", "tutor", "mentor connectivity"], {}),
        "Language Learning": (["language learning", "english learning", "japanese", "korean", "vocabulary"], {}),
        "Exam": (["exam", "quiz", "examination", "online judge", "test management"], {}),
        "Dormitory": (["dormitory", "dorm"], {}),
        "Alumni": (["alumni"], {}),
        "Academic Management": (["academic management", "student management", "training point", "student club"], {}),
    }),
    "Retail": (["retail", "shop"], {
        "Store": (["store", "shop", "retail store"], {}),
        "Marketplace": (["marketplace", "e-commerce", "ecommerce", "online market"], {}),
        "Inventory": (["inventory", "stock management"], {}),
        "Product Catalog": (["product catalog", "catalogue", "product listing"], {}),
        "Fashion": (["fashion", "clothing", "outfit", "uniform", "costume"], {}),
        "Second-hand": (["second-hand", "secondhand", "used products"], {}),
        "Print-on-Demand": (["print-on-demand", "pod", "custom artwork"], {}),
        "Bookstore": (["bookstore", "book commerce"], {}),
    }),
    "Food & Beverage": (["food", "beverage"], {
        "Meal Service": (["meal", "school lunch", "canteen meal"], {}),
        "Food Delivery": (["food delivery"], {}),
        "Nutrition": (["nutrition", "diet"], {}),
        "Seafood Supply": (["seafood", "seafood supply"], {}),
    }),
    "Finance": (["finance", "fintech"], {
        "Bank": (["bank", "banking", "core banking"], {}),
        "Wallet": (["wallet", "e-wallet", "payment wallet"], {}),
        "Loan": (["loan", "lending", "credit"], {}),
        "Personal Finance": (["personal finance", "expense", "bill splitting"], {}),
        "Stock Trading": (["stock", "stock trading", "paper trading"], {}),
    }),
    "Transportation": (["transportation", "mobility"], {
        "Parking": (["parking", "parking lot"], {}),
        "Ride-hailing": (["ride-hailing", "ride sharing", "taxi booking", "personal driver"], {}),
        "Bicycle Rental": (["bicycle rental", "bike rental", "bike sharing"], {}),
        "Bus": (["bus", "bus tracking", "public transport"], {}),
        "Freight Transport": (["freight", "pickup truck", "mass transportation", "cargo transport"], {}),
        "Car Rental": (["car rental", "vehicle rental", "self-driving rental"], {}),
        "Driving School": (["driving training", "driving school"], {}),
    }),
    "Real Estate": (["real estate", "realestate"], {
        "Apartment": (["apartment", "condo", "property operation"], {}),
        "Property Listing": (["property listing", "real estate listing", "property portal", "rental listing"], {}),
        "Roommate Finder": (["roommate", "roommate finder", "spare room"], {}),
    }),
    "Human Resources": (["human resources", "hr"], {
        "Employee": (["employee", "staff", "employee management"], {}),
        "Recruitment": (["recruitment", "hiring", "cv screening", "job board", "applicant tracking", "cv"], {}),
        "Payroll": (["payroll", "salary", "wage"], {}),
        "Freelancer": (["freelancer", "gig", "freelance", "bidding job"], {}),
        "Internship": (["internship", "ojt", "on-the-job training"], {}),
    }),
    "Government": (["government", "public sector"], {
        "Citizen": (["citizen", "resident"], {}),
        "License": (["license", "licensing", "certificate"], {}),
        "Permit": (["permit", "permit management", "approval"], {}),
        "Legal Compliance": (["legal", "legal compliance", "law"], {}),
    }),
    "Entertainment": (["entertainment"], {
        "Movie": (["movie", "cinema", "film"], {}),
        "Event": (["event", "event management", "conference"], {}),
        "Ticket": (["ticket", "ticketing", "booking ticket"], {}),
        "Game": (["game", "rpg", "roguelite", "unity", "photon", "detective game"], {}),
        "Music": (["music", "music collaboration", "music producer"], {}),
    }),
    "Agriculture": (["agriculture"], {
        "Farm": (["farm", "farming", "farm management"], {}),
        "Crop": (["crop", "crop management", "harvest"], {}),
        "Smart Garden": (["smart garden", "garden monitoring", "greenhouse"], {}),
    }),
    "Logistics": (["logistics"], {
        "Warehouse": (["warehouse", "depot", "storage"], {}),
        "Shipment": (["shipment", "freight", "cargo"], {}),
        "Courier Delivery": (["delivery", "last mile", "courier", "delivery locker"], {}),
        "Supply Chain": (["supply chain", "orchestration", "procurement"], {}),
    }),
    "Sports & Fitness": (["sports", "fitness"], {
        "Gym": (["gym", "fitness center"], {}),
        "Sports Field": (["sports field", "field booking", "coach"], {}),
    }),
    "Home Services": (["home service"], {
        "Repair Service": (["repair", "maintenance service", "home repair"], {}),
        "Construction": (["construction", "construction supervision"], {}),
        "Moving Service": (["moving", "furniture moving"], {}),
        "Domestic Help": (["domestic helper", "housekeeping"], {}),
    }),
    "Environment & Energy": (["environment", "energy"], {
        "Waste Management": (["waste", "waste monitoring"], {}),
        "Weather & Disaster": (["weather", "flood monitoring", "disaster"], {}),
        "Emission & Power": (["greenhouse gas", "emission", "thermal power"], {}),
        "Seaport Operations": (["seaport", "port operations"], {}),
    }),
    "Social & Community": (["social", "community"], {
        "Social Network": (["social network", "connection platform"], {}),
        "Family Tree": (["family tree", "genealogy"], {}),
        "Lost & Found": (["lost and found", "lost & found"], {}),
        "Charity": (["charity", "volunteering", "rescue"], {}),
    }),
    "Automotive": (["automotive"], {
        "Garage": (["garage", "auto repair"], {}),
        "Vehicle Service": (["vehicle service", "car maintenance"], {}),
    }),
    "Smart Home & IoT": (["smart home", "iot", "internet of things"], {
        "Home Automation": (["home automation", "smart device", "raspberry pi", "edge computing"], {}),
        "Delivery Locker": (["smart locker", "delivery locker"], {}),
    }),
    "Project & Work Management": (["project management", "work management"], {
        "Task Management": (["task management", "task assignment"], {}),
        "Business Analysis": (["business analyst", "requirement clarification"], {}),
    }),
}

# ── Layer 3: Methodology ────────────────────────────────────────────────────────────
METHOD = {
    "Agile": (["agile", "agile method"], {}),
    "Scrum": (["scrum", "sprint"], {}),
    "Waterfall": (["waterfall", "waterfall model"], {}),
    "Kanban": (["kanban", "kanban board"], {}),
    "TDD": (["tdd", "test driven"], {}),
    "Microservices": (["microservices", "microservice"], {}),
    "Monolith": (["monolith", "monolithic"], {}),
    "MVC": (["mvc", "model view controller"], {}),
    "Clean Architecture": (["clean architecture", "domain driven design", "ddd", "cqrs"], {}),
    "SaaS": (["saas", "software as a service", "multi-tenant"], {}),
}

# ── Layer 4: Task Type ──────────────────────────────────────────────────────────────
TASK = {
    "CRUD Management": (["crud management", "crud", "management system"], {}),
    "Booking": (["booking", "reservation", "appointment"], {}),
    "Recommendation": (["recommendation", "recommender", "suggestion"], {}),
    "Analytics": (["analytics", "reporting", "dashboard", "insight"], {}),
    "Authentication": (["authentication", "login", "auth"], {}),
    "Tracking & Monitoring": (["tracking", "monitoring", "real-time tracking"], {}),
    "Matching & Connecting": (["matching", "connecting", "connect"], {}),
    "Prediction & Forecasting": (["prediction", "forecasting", "risk calculation"], {}),
    "Chatbot & Assistant": (["chatbot", "virtual assistant", "ai assistant"], {}),
    "Search": (["search", "search engine"], {}),
    "Notification": (["notification", "alert", "reminder"], {}),
    "Payment Processing": (["payment", "online payment", "checkout", "vnpay", "payos", "momo"], {}),
}

LAYERS = [
    ("TechnicalStack", "technical_stack", "Technical Stack", TECH),
    ("DomainEntity", "domain_entity", "Domain Entity", DOMAIN),
    ("Methodology", "methodology", "Methodology", METHOD),
    ("TaskType", "task_type", "Task Type", TASK),
]


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _walk(concepts, used_ids, tree, parent_id, layer):
    for name, (keywords, children) in tree.items():
        cid = _slug(name)
        while cid in used_ids:
            cid += "_x"
        used_ids.add(cid)
        concepts.append({"id": cid, "name": name, "layer": layer, "parent": parent_id, "keywords": keywords})
        _walk(concepts, used_ids, children, cid, layer)


def build() -> dict:
    concepts = [{"id": "thing", "name": "Thing", "layer": "root", "parent": None, "keywords": []}]
    used_ids = {"thing"}
    for layer, root_id, root_name, tree in LAYERS:
        concepts.append({"id": root_id, "name": root_name, "layer": layer, "parent": "thing", "keywords": []})
        used_ids.add(root_id)
        _walk(concepts, used_ids, tree, root_id, layer)

    keyword_total = sum(len(c["keywords"]) for c in concepts)
    parents = {c["parent"] for c in concepts if c["parent"]}
    return {
        "_meta": {
            "status": "GENERATED by tools/build_sedo.py — edit the SPEC there, not this file.",
            "description": "Software Engineering Domain Ontology (SEDO), 4-layer hierarchy for DASSF. "
                           "English-only keywords; only isA (parent) is used by the measures.",
            "counts": {
                "concepts": len(concepts),
                "parent_classes": len(parents),
                "keywords": keyword_total,
                "domain_leaves": sum(1 for c in concepts if c["layer"] == "DomainEntity" and c["id"] not in parents),
                "tech_leaves": sum(1 for c in concepts if c["layer"] == "TechnicalStack" and c["id"] not in parents),
            },
        },
        "concepts": concepts,
    }


def main() -> None:
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "ontology", "sedo.json")
    data = build()
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}")
    print(json.dumps(data["_meta"]["counts"], indent=2))


if __name__ == "__main__":
    main()
