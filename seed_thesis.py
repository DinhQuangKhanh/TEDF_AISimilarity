"""Seed the Python similarity corpus from the web system's load-test topics.

Every thesis id equals the matching web project id (pairwise), so a "check duplicates"
run on a web project lines up with a row here. Titles are the English topic names.

The data is generated from ``backend/TEDF.Persistence/Seeds/LoadTestDataSeeder.cs``:
  - Fall 2025 topic i (0-based)  -> ProjectId(i + 1)            = 60000000-...-{i+1:012}
  - Spring 2026 topic i          -> ProjectId(50 + i + 1)       = 60000000-...-{51..90}
  - Summer 2026 group g          -> RealProjectId(g + 1)        = 61000000-...-{g+1:012}

Run once (uses ALEMBIC_DATABASE_URL / DATABASE_URL from .env, same as alembic):
    python seed_thesis.py
"""

import os
import uuid

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.thesis import Thesis
from app.services.similarity_service import SimilarityService

# Web project-id formulas, mirrored from LoadTestDataSeeder.cs verbatim.
FALL25_GROUP_COUNT = 50


def project_id(i: int) -> uuid.UUID:
    return uuid.UUID(f"60000000-0000-0000-0000-{i:012d}")


def real_project_id(i: int) -> uuid.UUID:
    return uuid.UUID(f"61000000-0000-0000-0000-{i:012d}")


FALL_2025 = [
    "Building a School Bus Management System for educational institutions using ReactJS, and ASP.NET",
    "Building digital platform that enables transactions between flower farms and flower shops using ReactJS, Spring Boot and MySQL",
    "Building a Fashion E-commerce Website for a Brand with Integrated AI to Optimize User Experience",
    "FamTree - Family Tree Management System",
    "Building a Tour Booking Management System for Korean Tourists in Da Nang Using Spring Boot, ReactJS, Android, and MySQL",
    "GaragePro-Building a digital garage management system to optimize the process of receiving, repairing, supporting incidents and delivering vehicles integrated on Web and Mobile platforms using ASP.NET WEB CORE API, Android, NextJS, SQL Server technology",
    "Building a web-based for renting travel equipment and essentials such as suitcases, cameras, and camping gear using ReactJS, NodeJS, and MongoDB",
    "EduMeal: Building A Web-based School Lunch Meal Management System using NextJs, ASP.NET Core Web API and SQL Server",
    "Building a System for Class Enrollment and Tracking at an English Center using ReactJS and ASP.NET Core Web API",
    "Building the Book Platform: Combining Book Commerce and AI Text-to-Speech Using ASP.NET Core API, React JS, and SQL Server Database",
    "Dozu - Personalized Learning Roadmaps Platform with Multi-Method Learning and Integrated Class Management System using Next.js, Node.js, PostgreSQL, Redis",
    "Building a system to connect, support, monitor the health and psychology of the elderly integrated on Web and Mobile platforms using NodeJS, ReactJS, MongoDB and React Native",
    "Building a system to support management of scientific research and articles in universities using ReactJS, Spring Boot and Mysql",
    "Smart Gym Management System Using NodeJS and React",
    "Building a web-based workspace safety management and operation system for construction sites using Node.js, React Js and MongoDB",
    "An online tutoring platform connecting tutors and students using AI for lecture content moderation and analysis using .NET and ReactJS",
    "HomeCareDN - Building a Construction and Repair Service Management System in Da Nang using ASP.NET Core API,ReactJS and SQL Server technology",
    "Building Event Management System in FPT University using ASP.NET Core Web API, ReactJS and SQL Server",
    "Building a comprehensive learning management and monitoring platform for training centers with React, ReactNative, Node.js and MongoDB",
    "Online Construction Supervision Platform for Residential Projects in Da Nang using Next.js, ASP.NET Core, PostgreSQL and Langchain",
    "Building a Smart Tutor-Student Matching and Learning Support Platform using REACT JS, EXPRESS JS, MONGODB",
    "Building a Microservices-Based Website for Managing the Supply Chain of Electronic Components and Microchip Devices using Spring Boot REST API, MySQL",
    "Roommate Finder Management System for Students in Da Nang City using NextJS, Java Spring Boot, PostgreSQL, and MongoDB",
    "Building an Online Platform for Student Connection and School Activity Management using ASP.NET Core Web API, ReactJS, and SQL Server",
    "Building Rentzy - Self-driving Vehicle Rental Platform using Node.js, React.js, and MySQL",
    "Building a website to support IT lecturers in organizing and managing course projects at FPT University Danang using microservice pattern",
    "Online platform for buying and selling smart homes integrated with IoT/ICT technology",
    "Faise Paper Trading - Real-Time Stock Trading Platform for Web and Mobile using Node.js, Python, React Native, MongoDB, MySQL and Redis",
    "Building ALLEN - A Platform supporting English learning using NextJS, ASP.NET Core API and SQL Server",
    "Building a Real-Time Public Waste Monitoring System Using IoT-Based Fill-Level Sensors Integrated with Web and Mobile Platforms via IoT Sensors, Spring Boot MVC, SQL Server and Android",
    "Build a luxury outfit rental and sales system, using Razor Pages, ASP.NET Core Web API and SQL Server",
    "Building a Co-working Space Booking System using ReactJS, ASP.NET Core Web API and SQL Server",
    "Building an online medical appointment booking platform using ASP.NET Core API, ReactTS, SQL Server technology",
    "Building a Web Application for Managing Seafood Supply and Consumption in Da Nang using ASP.NET Core API,ReactJS and SQL Server technology",
    "Building a data visualization website for helpful insight using ReactJs, NestJs and PostgreSQL",
    "Building an Apartment/House Rental Management System (Web and Mobile) using Spring Boot MVC, ReactJS, Firebase Database, Android",
    "FlexJob Connect: A platform that connects students and freelancers with job opportunities through bidding and contest-based mechanisms, built with Java Spring MVC, RESTful API and PostgreSQL",
    "Build an online website to book sports fields and find coaches using React, Nest Js, MongoDB",
    "Building a personal financial management system and dividing bills multi-platform multi-language using NextJS, React Native, Java Spring Boot technology",
    "Building EduXtend - A Student Club and Training Point Management System for FPT University, using ASP.NET Core Web API, ReactJS and SQL Server",
    "Building an E-Commerce Website for Second-Hand Products with Price Prediction AI using ReactJS, SQL Server and .NET",
    "Building a website system to support finding domestic helpers using NodeJS, SQL Server, ReactJS technology",
    "Building a Comprehensive Project Management and Music Collaboration Platform for Music Producers using Java Spring RESTful API, ReactJS and MySQL",
    "Building a Management System for Eco-Tourism Service Chain in Da Nang City using .NET, React, SQL Server Technology",
    "Build a Personalized Learning Website Using the FSRS Algorithm with NodeJS, ReactJS and MongoDB",
    "Build an AI-powered job board management system using NextJS, .NET Core and SQL Server",
    "Building an Event Ticketing Platform with React, Node.js, PostgreSQL and MongoDB",
    "Build a digital data portal on Vietnamese traditional festivals and beliefs using ASP.NET Core Web API, ReactJS and SQL Server",
    "Build LittleEdu - A Preschool Management System using React.js, ASP.NET RESTful API and PostgreSQL",
    "Building a Free Smart Online Learning System using React, Node.js and MySQL",
]

SPRING_2026 = [
    "Build a system to provide mass transportation services, connecting pickup truck drivers to users, using ReactJS, NodeJs, ReactNative, MongoDB",
    "Building a website to manage rescues and volunteering work using REACT JS, NODE JS, MONGODB technology",
    "GearXpert - An Online Smart Platform for Personal Electronics Rental, Automated Maintenance, and Intelligent Management using ReactJS, NodeJS, and MongoDB",
    "Building a restaurant management website system using REACT, ASP.NET and SQL Server",
    "FPT University Project Management System Using ReactJS, ASP.NET CORE WEB API, SQL Server, Firebase",
    "Building a system to support studying and practicing for Korean certificate exams for Vitamin Korean Language Center, using React, Spring Boot and MySQL",
    "Developing TikoSmart - a frozen food warehouse management system for TIKOVIA Trading and Service Co., Ltd. using ReactJS, React Native and NodeJS",
    "Building a web platform for purchasing, exchanging and selling bicycles and bicycle accessories using Next.JS technology and Java Spring Boot, PostgreSQL",
    "Building a website to promote and manage the Robotics, Chips and Emerging Technologies Lab of FPT University - Danang Campus, using React, Node.js (Express), and SQL Server",
    "Develop FigiCore - A retail and operational management system for collectible models using React.js, Nest.js and PostgreSQL",
    "Building a Household Furniture Moving Management System using ReactJS, NodeJS, and MongoDB",
    "Building a Small and Medium Enterprise Resource Planning System using ReactJS, ASP.NET Core Web API and Microsoft SQL Server",
    "Developing the EduConnect System - An AI-Integrated Educational Ecosystem for Learning, Testing, and Academic Discussion using VueJS, ASP.NET, and MySQL",
    "Building Online interview practice support System using ASP.NET Core Web API, ReactJS, PostgreSQL",
    "Petties: Veterinary Appointment Booking and AI-Powered Pet Disease Diagnosis System",
    "GZMart: An AI-Powered E-Commerce and Mini-ERP Platform for Fashion Retailers Using ReactJS, NodeJS, and MongoDB",
    "Building an RPG game using Unity with NPC interaction through AI and player support based on local data, utilizing Unity UI, ASP.NET APIs, and SQLServer technologies",
    "Building a management system for driving training centers in Da Nang city using NextJS, .NET C#, SQLServer, and MongoDB technologies",
    "Online dual-mode roguelite game design with AI assistance and Photon Fusion 2",
    "Developing a Nutrition and Exercise Tracking Mobile Application Using React Native (Expo), Express.js, and MongoDB",
    "Building a Student Dormitory Management System at FPT University Danang using ReactJS, NodeJS and MongoDB",
    "Building an AI-integrated recruitment platform for CV analysis and job recommendations using .NET 8 Web API, ReactJS, and MongoDB",
    "Developing the RestX System: A Restaurant Business Management Platform Using ReactJS, ASP.NET Core, and SQL Server",
    "An AI-integrated sports field booking and management system for venue owners on web and mobile platforms",
    "DOCIMAL AI - AI Agent Chatbot and Automation Platform - SaaS Product using Next.js, NestJS, FastAPI Microservices and RAG-LLM Technology",
    "Building a Virtual Try-On Platform for Student Uniform E-Commerce with ASP.NET Core Web API, Razor Pages, and SQL Server",
    "Building the StudySense - an AI-based system for learning style analysis and personalized self-study optimization using .NET, MySQL, Next.js",
    "ThemisOnlineJudge: Building a Web-based Online Programming Judge and Evaluation using NextJs, ASP.NET Core WebAPI and PostgreSQL",
    "Building a Cultural Experience and Craft Village Tourism Ecosystem in Ngu Hanh Son Ward Using React, Spring Boot and MySQL",
    "Building a smart library ecosystem SmartLib with HCE application, smart booking, reputation score and AI analysis using Flutter, PostgreSQL, Spring Boot",
    "Academic Management System at FPT University using Spring Boot, ReactJS, Flutter, and Python along with AI",
    "Developing an Intelligent Examination Room Management System Using NestJS and AI at FPT University",
    "Building a Personalized Travel Planning Platform using ReactJS, ASP.NET Core API, and PostgreSQL",
    "Developing a Web/App Platform to Support Interview Practice and Career Preparation by Industry using Next JS, .Net Core, PostgreSQL",
    "Building a Intelligent School Management System using React JS, Tailwind, Spring Boot, and PostgreSQL",
    "Website to diagnose Hand, Foot and Mouth Disease in Young Children through Images and Symptoms using AI, Machine Learning, ASP.NET Core, RESTful API and SQL Server",
    "Building an Integrated Lab Management System with Scheduling and Usage Tracking for FPT University Da Nang using ASP.NET API, ReactJS, and PostgreSQL",
    "Real-time Flood Monitoring and Safe Route Suggestion System using NextJS, React Native, ASP.NET Core, PostgreSQL",
    "Developing a Web Platform for Gym Management with Franchise and Shared Trainer Model using NodeJs and React",
    "Building a used car sales system in Da Nang city, using React, NextJS, Java Spring framework, SQL Server",
]

SUMMER_2026 = [
    "Task management by AI using model for suggesting and Risk calculations",
    "Multichannel Customer Feedback & Sentiment Analysis Hub",
    "GigBridge – An Intelligent AI Platform Connecting Freelancers and Businesses",
    "FUOJT – Developing a semester-based On-the-Job Training (OJT) management system with data separation and recruitment process tracking at FPT University Da Nang",
    "Building an Evaluation Framework for Detecting Duplicate Thesis Topics Based on Knowledge Domain Awareness in Software Engineering at FPT University Da Nang",
    "Developing a management and support system for establishing waterway tourism tours in Da Nang city",
    "Develop BusDN – a real-time bus management and tracking system in Da Nang City",
    "Intelligent Automotive Garage Management and Operating System",
    "Develop SafeRide – a real-time personal driver booking and management system",
    "PulmoCare – Pulmonary Disease Care and Monitoring System",
    "Building an E-Commerce Website for Second-Hand Products with Price Prediction AI using ReactJS, SQL Server, and .NET",
    "Building a management system for traditional medicine pharmacies in Da Nang using .NET, SQL Server, ReactJS technology.",
    "Develop a decision support and risk warning system for seaport operations based on real-time weather data",
]


def _rows():
    """Yields (thesis_id, semester, english_title) with ids matching the web projects."""
    for i, title in enumerate(FALL_2025):
        yield project_id(i + 1), "Fall 2025", title
    for i, title in enumerate(SPRING_2026):
        yield project_id(FALL25_GROUP_COUNT + i + 1), "Spring 2026", title
    for g, title in enumerate(SUMMER_2026):
        yield real_project_id(g + 1), "Summer 2026", title


def main() -> None:
    load_dotenv()
    db_url = os.getenv("ALEMBIC_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        raise SystemExit("Set DATABASE_URL (or ALEMBIC_DATABASE_URL) before seeding.")

    engine = create_engine(db_url)
    session = sessionmaker(bind=engine)()
    try:
        existing = {row[0] for row in session.query(Thesis.thesis_id).all()}
        new_ids: list[uuid.UUID] = []
        for thesis_id, semester, title in _rows():
            if thesis_id in existing:
                continue
            session.add(
                Thesis(thesis_id=thesis_id, semester=semester, program="SE", title=title)
            )
            new_ids.append(thesis_id)

        session.flush()
        if new_ids:
            # Pre-compute pairwise similarity so the first "check duplicates" is instant.
            SimilarityService(session).run_for_new(new_ids)
        session.commit()
        print(f"Seeded {len(new_ids)} theses (skipped {len(existing)} existing); similarity computed.")
    finally:
        session.close()


if __name__ == "__main__":
    main()
