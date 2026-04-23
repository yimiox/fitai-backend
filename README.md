# EvidenceFit

EvidenceFit is an evidence-based, AI-powered health and fitness application. It generates personalized workout and meal plans tailored to the user's goals, body metrics, and available equipment, backing its recommendations with citations from scientific research. It also features an AI coach that provides evidence-based answers to user queries.

## Features
- **Personalized Planning:** Creates tailored workout routines and meal plans based on your profile (age, weight, goal, diet preference, training hours).
- **Vision AI Equipment Inventory:** Upload a photo of your gym setup or home equipment. The AI identifies your equipment and builds a routine exclusively using what you have.
- **Evidence-Based Rationale:** Utilizes **Retrieval-Augmented Generation (RAG)** to provide citations and research sources justifying the structure of your plan. Every meaningful claim is cited from the indexed corpus or marked as general guidance.
- **AI Coach Chat:** A chat interface where you can ask fitness and nutrition questions. It responds using scientific literature and provides direct citations.

## Safety and Security 
- **Input Sanitization:** All user input is rigorously sanitized and screened for jailbreak patterns to prevent malicious exploits.
- **Guardrails:** Robust system-level constraints prevent prompt injection and ensure advice remains strictly within the realm of fitness and nutrition.
- **Privacy:** All conversation memory is private and scoped to the users only, ensuring that personal health journeys remain confidential and accessible only to the account owner.

## How It Works

The application flow leverages React on the frontend and Supabase (Auth, Database, and Edge Functions) on the backend to orchestrate AI generation.

```mermaid
graph TD
    A([User Sign Up / Log In<br>Supabase Auth]) --> B{Profile Completed?}
    
    B -- No --> C[Onboarding<br>Enter metrics, goals, diet]
    C --> D[(Save to 'profiles' table)]
    D --> E[[Generate Plan<br>Edge Function]]
    
    B -- Yes --> F[Dashboard]
    E --> F
    
    F --> G[View Workout & Meals<br>Tabs: Workout, Meals, Sources]
    
    F --> H[Regenerate Plan]
    H -.->|Optional| I[Upload Equipment Image<br>Vision AI processing]
    I --> E
    H --> E
    
    F --> J[Ask the Coach<br>AI Chat Interface]
    J --> K[[Chat Edge Function<br>Streaming SSE]]
    K --> L[Response + Citations]
    
    style E fill:#e1d5e7,stroke:#9673a6,stroke-width:2px
    style K fill:#e1d5e7,stroke:#9673a6,stroke-width:2px
    style D fill:#dae8fc,stroke:#6c8ebf,stroke-width:2px
```

### 1. Onboarding & Profiling
New users start at the onboarding screen to fill out basic stats (gender, age, height, weight, activity level) and their goal (e.g., build muscle, lose fat). This data is stored securely in the Supabase PostgreSQL `profiles` table.

### 2. Plan Generation
Once onboarded, a request is made to the `generate-plan` Supabase Edge Function. It calls the **llama-3.3-70b-versatile** model to generate a customized JSON structure representing the user's workout split, exercises (with sets, reps, and RIR), and a daily meal plan with macro breakdowns.

### 3. Equipment Vision Tailoring (Optional)
On the Dashboard, users can upload an image of their gym or available equipment. The frontend sends this base64 image to the `generate-plan` function, which uses the **meta-llama/llama-4-scout-17b-16e-instruct** to identify the available equipment and constraints the workout plan accordingly.

### 4. AI Coach Chat
Users can interact with an AI coach via the `/chat` route. Messages are sent to the chat Edge Function, which streams back responses using Server-Sent Events (SSE). The AI leverages RAG to focus on providing scientific rationale and specific peer-reviewed citations.

## Tech Stack
- **Frontend:** React, Vite, Tailwind CSS, Radix UI components (shadcn/ui), Framer Motion.
- **Backend & Auth:** Supabase (PostgreSQL, Authentication).
- **Serverless / AI:** Supabase Edge Functions, Groq API (llama models).
- **Routing:** React Router DOM.
- **Icons & Markdown:** Lucide React, React Markdown (with remark-gfm).
