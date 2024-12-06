from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from pathlib import Path

# Path ของโมเดลที่ฝึกไว้
MODEL_DIR = Path("../../models")
model_name_or_path = str(MODEL_DIR / "results/checkpoint-37674")  # แปลง Path เป็น String

# Mapping ระหว่าง id และ label (ตามการฝึก)
id2label = {
    0: "O",
    1: "B-ROLE",
    2: "I-ROLE",
    3: "B-SKILL",
    4: "I-SKILL",
    5: "B-TECH",
    6: "I-TECH",
}
label2id = {v: k for k, v in id2label.items()}

# โหลดโมเดลและบังคับตั้งค่า id2label และ label2id
model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# กำหนด id2label และ label2id ให้โมเดล
model.config.id2label = id2label
model.config.label2id = label2id

# สร้าง pipeline
token_classifier = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",  # รวม subwords เป็นคำเดียว
    device=0,  # ใช้ GPU (ถ้าไม่มี CUDA ให้เปลี่ยนเป็น -1)
)


# ฟังก์ชันโหลดข้อความจากไฟล์
def load_test_sentences(file_path):
    """โหลดข้อความทดสอบจากไฟล์ .txt"""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as file:
        sentences = file.readlines()
    return [sentence.strip() for sentence in sentences if sentence.strip()]


# ฟังก์ชันทดสอบข้อความ
def test_model_on_sentences(sentences, classifier):
    """ทดสอบโมเดลด้วยข้อความหลายประโยค"""
    for idx, sentence in enumerate(sentences, start=1):
        print(f"\n[{idx}] Input Sentence: {sentence}")
        results = classifier(sentence)
        if results:
            print("Extracted Entities:")
            for result in results:
                print(
                    f"Entity Group: {result['entity_group']}, Word: '{result['word']}', Confidence Score: {result['score']:.4f}"
                )
        else:
            print("No entities detected.")


# ตัวเลือก: เก็บข้อความในไฟล์หรือในตัวแปร Python
USE_FILE = False  # หากต้องการใช้ไฟล์ ตั้งค่าเป็น True

if USE_FILE:
    # กรณีโหลดข้อความจากไฟล์
    TEST_SENTENCES_FILE = Path("test_sentences.txt")  # ชื่อไฟล์ข้อความ
    test_sentences = load_test_sentences(TEST_SENTENCES_FILE)
else:
    # กรณีใช้ข้อความจากตัวแปร Python
    test_sentences = [
        "1. We are hiring a software engineer with expertise in Java, Kubernetes, and Docker.",
        "2. The ideal candidate should be proficient in data analysis, machine learning, and deep learning.",
        "3. Looking for a cybersecurity analyst familiar with penetration testing and ethical hacking.",
        "4. Our company is seeking a front-end developer skilled in HTML, CSS, and JavaScript frameworks like React.",
        "5. We need a backend engineer experienced in Node.js, Express, and MongoDB.",
        "6. A project manager with strong skills in Agile methodologies and team leadership is required.",
        "7. The data scientist should have experience working with Python, R, and SQL for data wrangling and visualization.",
        "8. Hiring a DevOps engineer capable of managing CI/CD pipelines and cloud infrastructure on AWS.",
        "9. The software architect must understand system design and have experience with microservices architecture.",
        "10. We are searching for an AI researcher with knowledge of natural language processing (NLP) and computer vision.",
        "11. The QA engineer should have experience in automated testing using Selenium and JUnit.",
        "12. A cloud engineer proficient in Azure, Terraform, and serverless technologies is needed.",
        "13. Seeking a mobile developer skilled in Kotlin for Android and Swift for iOS development.",
        "14. A machine learning engineer with a solid understanding of TensorFlow, PyTorch, and feature engineering is required.",
        "15. We are looking for a blockchain developer to work on Ethereum smart contracts using Solidity.",
        "16. Hiring a business analyst experienced in requirements gathering, stakeholder communication, and process modeling.",
        "17. A network engineer with expertise in routing protocols like BGP and OSPF is needed.",
        "18. Searching for a game developer familiar with Unity or Unreal Engine and C# programming.",
        "19. The technical writer should have experience creating API documentation and technical guides.",
        "20. Looking for a UX/UI designer proficient in Figma, Sketch, and user research methodologies.",
        "21. Wanna hire a Python guru who knows Django and Flask like the back of their hand.",
        "22. Need a full-stack developer who can juggle React, Node.js, and GraphQL with ease.",
        "23. Seeking an ML ops engineer to deploy machine learning models on Kubernetes clusters.",
        "24. Hiring a SQL expert for building optimized queries and maintaining data warehouses.",
        "25. Searching for a VR developer with hands-on experience in Oculus SDK or ARKit.",
        "26. Need a network security specialist who can configure firewalls and monitor IDS/IPS systems.",
        "27. We need someone who's a pro in AWS Lambda, DynamoDB, and API Gateway for our serverless app.",
        "28. Hiring a database admin familiar with PostgreSQL replication and backup strategies.",
        "29. Looking for an AI engineer who knows GPT fine-tuning and prompt engineering.",
        "30. The ideal tech lead should be comfortable mentoring junior developers and conducting code reviews.",
    ]

# เรียกใช้งานฟังก์ชันทดสอบ
test_model_on_sentences(test_sentences, token_classifier)
