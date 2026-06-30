---
title: Civic Ray
emoji: ⚖️
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: true
---

# ⚖️ CIVIC-RAY Municipal Law AI Protocol

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:000000,50:8B0000,100:FF0000&height=240&section=header&text=CIVIC%20RAY&fontSize=70&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Intelligent%20Municipal%20Bye-Laws%20Assistant&descAlignY=60&descSize=20" />

<br/>

<img src="https://readme-typing-svg.herokuapp.com?font=Orbitron&size=28&duration=3000&pause=1000&color=FF0000&center=true&vCenter=true&width=800&lines=Your+AI+Counsel+for+Municipal+Laws;Citizen+%26+Lawyer+Dual+Modes;RAG+Powered+Legal+Reasoning;Built+With+LangChain+%2B+FAISS" />

<br/><br/>

[![Live Demo](https://img.shields.io/badge/LIVE_APP-black?style=for-the-badge&logo=huggingface&logoColor=red)](https://huggingface.co/spaces/Abhinav23124/Civic-Ray)
[![Backend](https://img.shields.io/badge/Backend-Python%2FFlask-black?style=for-the-badge&logo=flask&logoColor=red)]()
[![Vector DB](https://img.shields.io/badge/VectorDB-FAISS-black?style=for-the-badge&logo=meta&logoColor=red)]()
[![LLM](https://img.shields.io/badge/LLM-Gemini%20%7C%20LLaMA-black?style=for-the-badge&logo=openai&logoColor=red)]()

</div>

---

# 🧠 System Overview

**CIVIC-RAY** is an intelligent Retrieval-Augmented Generation (RAG) platform designed to democratize access to Indian municipal bye-laws. It serves as an AI counsel that bridges the gap between complex legal jargon and actionable civic knowledge.

The platform provides:
- Semantic search across complex legal documents
- Smart query expansion using legal terminology
- Automated risk assessments for violations
- Generative drafting of formal municipal notices
- Dual-persona response tailoring
- Real-time grounded answers from official bye-laws

The system is engineered to be the ultimate civic-tech tool for citizens and legal professionals alike.

---

# ⚡ Core Features

## ⚖️ Dual Persona System

- **Citizen Mode** — Plain language responses
- **Lawyer Mode** — Formal legal drafting
- Real-Time Mode Switching
- Persona-Tailored Outputs
- Interactive Mode Toggle

---

## 🔍 Semantic Retrieval Protocol

- FAISS Vector Database
- Dense Embedding Search
- Top-K Document Retrieval
- Source Page Citations
- Sub-Second Response Time

---

## 🧠 LLM Fallback Engine

- Primary: Google Gemini 2.0 Flash
- Fallback: Meta LLaMA 3.3 70B
- Zero-Downtime Switching
- Multi-Provider Routing
- Quota-Aware Optimization

---

## 📈 Query Intelligence System

- Legal Keyword Expansion
- Colloquial-to-Formal Translation
- Auto Fine Detection (₹)
- Section Reference Extraction
- Smart Greeting Shortcut

---

## 📜 Legal Drafting Engine

- Risk Assessment Generator
- Formal Notice Builder
- Municipal Commissioner Templates
- Verbatim Section Citations
- Structured Output Parsing

---

## 🛡️ Hallucination Prevention

- Strict Context Bounding
- Grounded Generation Only
- No Invented Fine Amounts
- Source-Cited Responses
- Trusted Legal Output

---

# 🌐 Use Cases

- Illegal Construction Queries
- Wrongful Fine Disputes
- Pothole & Road Damage Claims
- Property Tax Violations
- Sanitation Bye-Laws
- Building Permit Issues

---

# 🔒 Platform Security

- Environment Variable Secrets
- Hugging Face Secret Manager
- Non-Root Docker User
- Input Sanitization
- Secure API Routing

---

# 🧰 Technology Stack

<div align="center">

<img src="https://skillicons.dev/icons?i=python,flask,docker,html,css,js,git,github,vscode" />
<br/><br/>
<img src="https://img.shields.io/badge/LangChain-black?style=for-the-badge&logo=langchain&logoColor=red" />
<img src="https://img.shields.io/badge/HuggingFace-black?style=for-the-badge&logo=huggingface&logoColor=red" />
<img src="https://img.shields.io/badge/FAISS-black?style=for-the-badge&logo=meta&logoColor=red" />
<img src="https://img.shields.io/badge/Gunicorn-black?style=for-the-badge&logo=gunicorn&logoColor=red" />

</div>

---

# 🏗️ System Architecture

```txt
User Interface Dashboard
            ↓
Query Optimization Engine
            ↓
Dense Embedding Encoder (mpnet)
            ↓
FAISS Vector Retrieval Layer
            ↓
Context Assembly Module
            ↓
LLM Generation Engine (Gemini → LLaMA)
            ↓
Structured Response Parser
            ↓
Citizen / Lawyer Output Renderer