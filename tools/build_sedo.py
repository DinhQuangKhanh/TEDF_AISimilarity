"""Generate ``app/ontology/sedo.json`` from a compact, human-editable spec.

SEDO (paper §V-D) is a four-layer SE ontology: Technical Stack · Methodology · Domain
Entity · Task Type. Only the ``isA`` (parent) relation is used by the similarity measures
in ``app/ontology/sedo.py`` (Wu-Palmer, wpath). Concepts carry surface ``keywords`` — both
English and Vietnamese — that Module-1 NER matches against a title.

The paper says SEDO "grows iteratively as the evaluation phase surfaces missing concepts",
so keeping the ontology as a spec + generator (rather than hand-edited JSON) makes that growth
cheap and keeps the file consistent. Edit ``SPEC`` below, then run:

    python tools/build_sedo.py

Spec format — a nested tree of ``name: (keywords, children)``:
    "React": (["react", "reactjs", "react.js"], {})           # leaf
    "Frontend": (["frontend", "giao diện"], { ...children... }) # parent
"""

from __future__ import annotations

import json
import os
import re

# ── Layer 1: Technical Stack ────────────────────────────────────────────────────────
TECH = {
    "Frontend": (["frontend", "front-end", "giao diện", "client side"], {
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
    "Backend": (["backend", "back-end", "server side", "máy chủ"], {
        "Node.js": (["node.js", "nodejs", "node js", "express", "express.js", "expressjs"], {}),
        "NestJS": (["nestjs", "nest.js", "nest js"], {}),
        "Spring Boot": (["spring boot", "spring", "java spring", "spring mvc"], {}),
        "ASP.NET Core": (["asp.net core", "asp.net", "dotnet core", ".net core", ".net 8", "web api", "aspnet"], {}),
        "Django": (["django", "django rest"], {}),
        "Flask": (["flask"], {}),
        "FastAPI": (["fastapi", "fast api"], {}),
        "Laravel": (["laravel", "php laravel", "php"], {}),
    }),
    "Database": (["database", "cơ sở dữ liệu", "csdl"], {
        "PostgreSQL": (["postgresql", "postgres", "psql", "pgvector"], {}),
        "MySQL": (["mysql", "mariadb"], {}),
        "MongoDB": (["mongodb", "mongo", "nosql", "mongoose", "mongodb atlas"], {}),
        "SQL Server": (["sql server", "mssql", "sqlserver"], {}),
        "Redis": (["redis", "redis cache"], {}),
        "Firebase": (["firebase", "firestore", "firebase database"], {}),
        "Supabase": (["supabase"], {}),
    }),
    "Mobile": (["mobile", "di động", "mobile app", "ứng dụng di động"], {
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
    "Integration": (["integration", "tích hợp"], {
        "REST API": (["rest api", "rest", "restful", "web service"], {}),
        "GraphQL": (["graphql", "graph ql"], {}),
        "gRPC": (["grpc"], {}),
        "Realtime": (["websocket", "signalr", "socket.io", "socketio", "realtime", "thời gian thực"], {}),
        "Message Queue": (["kafka", "rabbitmq", "message queue", "message broker"], {}),
    }),
    "AI/ML": (["ai", "artificial intelligence", "trí tuệ nhân tạo", "machine learning", "học máy"], {
        "Machine Learning": (["machine learning", "scikit-learn", "sklearn", "xgboost", "prediction model", "forecasting model"], {}),
        "Deep Learning": (["deep learning", "neural network", "pytorch", "tensorflow", "học sâu"], {}),
        "NLP": (["nlp", "natural language processing", "sentiment analysis", "text analysis"], {}),
        "Computer Vision": (["computer vision", "image recognition", "object detection", "thị giác máy tính"], {}),
        "LLM/RAG": (["llm", "rag", "langchain", "openai", "gemini", "chatbot", "generative ai", "large language model"], {}),
        "OCR": (["ocr", "tesseract", "optical character recognition"], {}),
    }),
}

# ── Layer 2: Domain Entity (business domains) ───────────────────────────────────────
DOMAIN = {
    "Hospitality": (["hospitality", "lưu trú"], {
        "Hotel": (["hotel", "khách sạn", "lodging"], {}),
        "Restaurant": (["restaurant", "nhà hàng", "dining", "canteen", "căng tin", "food ordering"], {}),
        "Homestay": (["homestay", "guest house", "nhà nghỉ"], {}),
        "Tourism": (["tourism", "travel", "du lịch", "tour", "itinerary", "waterway tourism", "eco-tourism"], {}),
        "Coworking Space": (["coworking", "co-working", "workspace booking", "không gian làm việc"], {}),
    }),
    "Healthcare": (["healthcare", "y tế", "medical", "chăm sóc sức khỏe"], {
        "Hospital": (["hospital", "bệnh viện"], {}),
        "Clinic": (["clinic", "phòng khám", "outpatient", "medical appointment", "đặt lịch khám"], {}),
        "Pharmacy": (["pharmacy", "nhà thuốc", "drugstore", "medicine", "medication"], {}),
        "Laboratory": (["laboratory", "lab", "diagnostics", "xét nghiệm"], {}),
        "Mental Health": (["mental health", "psychology", "sức khỏe tâm thần", "tâm lý"], {}),
        "Veterinary": (["veterinary", "pet", "thú y", "thú cưng"], {}),
        "Elderly Care": (["elderly", "elderly care", "người cao tuổi", "chăm sóc người già"], {}),
        "Disease Diagnosis": (["diagnosis", "disease detection", "chẩn đoán bệnh", "pulmonary", "lung disease"], {}),
    }),
    "Education": (["education", "giáo dục", "learning", "học tập", "e-learning"], {
        "School": (["school", "trường học", "k12", "preschool", "kindergarten", "mầm non"], {}),
        "University": (["university", "campus", "higher education", "đại học", "fpt university"], {}),
        "Course": (["course", "e-learning", "online learning", "lms", "khóa học", "học trực tuyến"], {}),
        "Library": (["library", "thư viện", "book borrowing"], {}),
        "Tutoring": (["tutoring", "tutor", "gia sư", "dạy kèm", "mentor connectivity"], {}),
        "Language Learning": (["language learning", "english learning", "japanese", "korean", "học ngoại ngữ", "vocabulary"], {}),
        "Exam": (["exam", "quiz", "examination", "thi cử", "online judge", "test management"], {}),
        "Dormitory": (["dormitory", "dorm", "ký túc xá"], {}),
        "Alumni": (["alumni", "cựu sinh viên"], {}),
        "Academic Management": (["academic management", "student management", "training point", "student club", "quản lý học vụ"], {}),
    }),
    "Retail": (["retail", "bán lẻ", "shop", "cửa hàng"], {
        "Store": (["store", "shop", "retail store"], {}),
        "Marketplace": (["marketplace", "e-commerce", "ecommerce", "online market", "thương mại điện tử", "sàn giao dịch"], {}),
        "Inventory": (["inventory", "stock management", "kho hàng", "tồn kho"], {}),
        "Product Catalog": (["product catalog", "catalogue", "product listing"], {}),
        "Fashion": (["fashion", "clothing", "outfit", "uniform", "thời trang", "quần áo", "costume"], {}),
        "Second-hand": (["second-hand", "secondhand", "used products", "đồ cũ", "hàng cũ"], {}),
        "Print-on-Demand": (["print-on-demand", "pod", "custom artwork", "in theo yêu cầu"], {}),
        "Bookstore": (["bookstore", "book commerce", "nhà sách", "hiệu sách"], {}),
    }),
    "Food & Beverage": (["food", "beverage", "thực phẩm", "đồ ăn"], {
        "Meal Service": (["meal", "school lunch", "canteen meal", "bữa ăn", "suất ăn"], {}),
        "Food Delivery": (["food delivery", "giao đồ ăn", "đặt món"], {}),
        "Nutrition": (["nutrition", "dinh dưỡng", "diet"], {}),
        "Seafood Supply": (["seafood", "hải sản", "seafood supply"], {}),
    }),
    "Finance": (["finance", "tài chính", "fintech"], {
        "Bank": (["bank", "banking", "ngân hàng", "core banking"], {}),
        "Wallet": (["wallet", "e-wallet", "payment wallet", "ví điện tử"], {}),
        "Loan": (["loan", "lending", "credit", "vay", "tín dụng"], {}),
        "Personal Finance": (["personal finance", "expense", "bill splitting", "quản lý chi tiêu", "chia hóa đơn"], {}),
        "Stock Trading": (["stock", "stock trading", "paper trading", "chứng khoán", "đầu tư"], {}),
    }),
    "Transportation": (["transportation", "giao thông", "vận tải", "mobility"], {
        "Parking": (["parking", "parking lot", "bãi đỗ xe", "đỗ xe"], {}),
        "Ride-hailing": (["ride-hailing", "ride sharing", "taxi booking", "personal driver", "gọi xe", "tài xế"], {}),
        "Bicycle Rental": (["bicycle rental", "bike rental", "bike sharing", "thuê xe đạp"], {}),
        "Bus": (["bus", "bus tracking", "public transport", "xe buýt", "giao thông công cộng"], {}),
        "Freight Transport": (["freight", "pickup truck", "mass transportation", "cargo transport", "vận chuyển hàng"], {}),
        "Car Rental": (["car rental", "vehicle rental", "self-driving rental", "thuê xe"], {}),
        "Driving School": (["driving training", "driving school", "đào tạo lái xe", "học lái xe"], {}),
    }),
    "Real Estate": (["real estate", "realestate", "bất động sản"], {
        "Apartment": (["apartment", "condo", "chung cư", "căn hộ", "property operation"], {}),
        "Property Listing": (["property listing", "real estate listing", "property portal", "rental listing"], {}),
        "Roommate Finder": (["roommate", "roommate finder", "tìm bạn ở ghép", "spare room"], {}),
    }),
    "Human Resources": (["human resources", "nhân sự", "hr"], {
        "Employee": (["employee", "staff", "nhân viên", "employee management"], {}),
        "Recruitment": (["recruitment", "hiring", "cv screening", "job board", "applicant tracking", "tuyển dụng", "cv"], {}),
        "Payroll": (["payroll", "salary", "wage", "lương"], {}),
        "Freelancer": (["freelancer", "gig", "freelance", "việc tự do", "bidding job"], {}),
        "Internship": (["internship", "ojt", "on-the-job training", "thực tập"], {}),
    }),
    "Government": (["government", "chính phủ", "public sector", "hành chính"], {
        "Citizen": (["citizen", "resident", "công dân", "cư dân"], {}),
        "License": (["license", "licensing", "certificate", "giấy phép", "chứng chỉ"], {}),
        "Permit": (["permit", "permit management", "approval", "phê duyệt"], {}),
        "Legal Compliance": (["legal", "legal compliance", "law", "pháp lý", "tuân thủ"], {}),
    }),
    "Entertainment": (["entertainment", "giải trí"], {
        "Movie": (["movie", "cinema", "film", "phim", "rạp chiếu"], {}),
        "Event": (["event", "event management", "conference", "sự kiện", "hội nghị"], {}),
        "Ticket": (["ticket", "ticketing", "booking ticket", "vé", "đặt vé"], {}),
        "Game": (["game", "rpg", "roguelite", "unity", "photon", "detective game", "trò chơi"], {}),
        "Music": (["music", "music collaboration", "âm nhạc", "music producer"], {}),
    }),
    "Agriculture": (["agriculture", "nông nghiệp"], {
        "Farm": (["farm", "farming", "farm management", "nông trại", "trang trại"], {}),
        "Crop": (["crop", "crop management", "harvest", "cây trồng", "mùa vụ"], {}),
        "Smart Garden": (["smart garden", "garden monitoring", "greenhouse", "vườn thông minh", "nhà kính"], {}),
    }),
    "Logistics": (["logistics", "hậu cần", "chuỗi cung ứng"], {
        "Warehouse": (["warehouse", "depot", "storage", "kho bãi", "nhà kho"], {}),
        "Shipment": (["shipment", "freight", "cargo", "lô hàng"], {}),
        "Courier Delivery": (["delivery", "last mile", "courier", "delivery locker", "giao hàng", "chuyển phát"], {}),
        "Supply Chain": (["supply chain", "chuỗi cung ứng", "orchestration", "procurement"], {}),
    }),
    "Sports & Fitness": (["sports", "fitness", "thể thao", "thể hình"], {
        "Gym": (["gym", "fitness center", "phòng gym", "phòng tập"], {}),
        "Sports Field": (["sports field", "field booking", "coach", "sân thể thao", "đặt sân", "huấn luyện viên"], {}),
    }),
    "Home Services": (["home service", "dịch vụ gia đình"], {
        "Repair Service": (["repair", "maintenance service", "home repair", "sửa chữa"], {}),
        "Construction": (["construction", "construction supervision", "xây dựng", "giám sát công trình"], {}),
        "Moving Service": (["moving", "furniture moving", "chuyển nhà", "dọn nhà"], {}),
        "Domestic Help": (["domestic helper", "housekeeping", "giúp việc", "người giúp việc"], {}),
    }),
    "Environment & Energy": (["environment", "energy", "môi trường", "năng lượng"], {
        "Waste Management": (["waste", "waste monitoring", "rác thải", "quản lý rác"], {}),
        "Weather & Disaster": (["weather", "flood monitoring", "disaster", "thời tiết", "lũ lụt", "cảnh báo thiên tai"], {}),
        "Emission & Power": (["greenhouse gas", "emission", "thermal power", "khí thải", "nhà máy điện"], {}),
        "Seaport Operations": (["seaport", "port operations", "cảng biển", "vận hành cảng"], {}),
    }),
    "Social & Community": (["social", "community", "cộng đồng", "xã hội"], {
        "Social Network": (["social network", "connection platform", "mạng xã hội", "kết nối"], {}),
        "Family Tree": (["family tree", "genealogy", "gia phả"], {}),
        "Lost & Found": (["lost and found", "lost & found", "đồ thất lạc"], {}),
        "Charity": (["charity", "volunteering", "rescue", "từ thiện", "tình nguyện", "cứu trợ"], {}),
    }),
    "Automotive": (["automotive", "ô tô", "xe hơi"], {
        "Garage": (["garage", "auto repair", "gara", "sửa xe"], {}),
        "Vehicle Service": (["vehicle service", "car maintenance", "bảo dưỡng xe"], {}),
    }),
    "Smart Home & IoT": (["smart home", "iot", "internet of things", "nhà thông minh"], {
        "Home Automation": (["home automation", "smart device", "raspberry pi", "edge computing", "thiết bị thông minh"], {}),
        "Delivery Locker": (["smart locker", "delivery locker", "tủ khóa thông minh"], {}),
    }),
    "Project & Work Management": (["project management", "quản lý dự án", "work management"], {
        "Task Management": (["task management", "task assignment", "quản lý công việc", "giao việc"], {}),
        "Business Analysis": (["business analyst", "requirement clarification", "phân tích nghiệp vụ"], {}),
    }),
}

# ── Layer 3: Methodology ────────────────────────────────────────────────────────────
METHOD = {
    "Agile": (["agile", "agile method"], {}),
    "Scrum": (["scrum", "sprint"], {}),
    "Waterfall": (["waterfall", "waterfall model"], {}),
    "Kanban": (["kanban", "kanban board"], {}),
    "TDD": (["tdd", "test driven"], {}),
    "Microservices": (["microservices", "microservice", "vi dịch vụ"], {}),
    "Monolith": (["monolith", "monolithic"], {}),
    "MVC": (["mvc", "model view controller"], {}),
    "Clean Architecture": (["clean architecture", "domain driven design", "ddd", "cqrs"], {}),
    "SaaS": (["saas", "software as a service", "multi-tenant", "đa người thuê"], {}),
}

# ── Layer 4: Task Type ──────────────────────────────────────────────────────────────
TASK = {
    "CRUD Management": (["crud management", "crud", "management system", "quản lý"], {}),
    "Booking": (["booking", "reservation", "appointment", "đặt lịch", "đặt chỗ"], {}),
    "Recommendation": (["recommendation", "recommender", "suggestion", "gợi ý", "đề xuất"], {}),
    "Analytics": (["analytics", "reporting", "dashboard", "báo cáo", "thống kê", "insight"], {}),
    "Authentication": (["authentication", "login", "auth", "đăng nhập", "xác thực"], {}),
    "Tracking & Monitoring": (["tracking", "monitoring", "real-time tracking", "theo dõi", "giám sát"], {}),
    "Matching & Connecting": (["matching", "connecting", "connect", "kết nối", "ghép nối"], {}),
    "Prediction & Forecasting": (["prediction", "forecasting", "risk calculation", "dự đoán", "dự báo"], {}),
    "Chatbot & Assistant": (["chatbot", "virtual assistant", "ai assistant", "trợ lý ảo"], {}),
    "Search": (["search", "search engine", "tìm kiếm"], {}),
    "Notification": (["notification", "alert", "reminder", "thông báo", "nhắc nhở"], {}),
    "Payment Processing": (["payment", "online payment", "checkout", "thanh toán", "vnpay", "payos", "momo"], {}),
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
                           "Bilingual (EN + VI) keywords; only isA (parent) is used by the measures.",
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
