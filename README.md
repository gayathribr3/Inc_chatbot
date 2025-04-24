# LIC-Bot

An AI chatbot for answering insurance-related queries using LIC policy data.

## Features

1. High-Speed Reasoning powered by DeepSeek R1, enabling rapid understanding of complex queries.
2. Ultra-Fast Inference using Groq, ensuring near-instant responses.
3. Robust LIC Knowledge Base trained specifically on LIC insurance policies, offering accurate and reliable answers.

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/gayathribr3/Inc_chatbot.git
cd Inc_chatbot
```

### 2. Make Sure Python is Installed

Install Python 3.10 if not already installed.

---

### 3. Create and Activate a Virtual Environment

```bash
conda create -n lic-bot python=3.10
conda activate lic-bot
```

---

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 5. Set Up Environment Variables

Generate a Groq API key from [https://console.groq.com](https://console.groq.com)  
Then create a `.env` file and add:

```
GROQ_API_KEY=your-api-key-here
```

---

### 6. Run the Bot

```bash
python app.py
```

---

## ✅ Ready to Answer Insurance Questions Only!

Example:

```txt
✅ What does LIC Jeevan Anand cover?
✅ How do I file a claim?
❌ What's your favorite movie?
```

---

```

```
