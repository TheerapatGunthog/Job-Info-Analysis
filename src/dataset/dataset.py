import pandas as pd
import re
from typing import List, Dict
import os

# เพิ่ม constants สำหรับ NER labels
id2label = {
    0: "O",  # Other
    1: "B-SKILL",  # Beginning of Skill
    2: "I-SKILL",  # Inside of Skill
    3: "B-TECHNOLOGY",  # Beginning of Technology
    4: "I-TECHNOLOGY",  # Inside of Technology
}
label2id = {v: k for k, v in id2label.items()}

# เพิ่ม keywords สำหรับแต่ละ entity
KEYWORDS = {
    "SKILL": {
        # Programming Skills
        "programming",
        "coding",
        "software development",
        "web development",
        "mobile development",
        "database design",
        "system design",
        "api development",
        "backend development",
        "frontend development",
        "full stack development",
        "software architecture",
        "system architecture",
        "microservices",
        "test driven development",
        "behavior driven development",
        "object oriented programming",
        "functional programming",
        # Data Skills
        "data analysis",
        "machine learning",
        "deep learning",
        "data mining",
        "statistical analysis",
        "data modeling",
        "data visualization",
        "predictive modeling",
        "natural language processing",
        "computer vision",
        "big data",
        "data warehousing",
        "etl",
        "data pipeline",
        "business intelligence",
        "data science",
        "analytics",
        "artificial intelligence",
        "neural networks",
        "time series analysis",
        # Cloud & DevOps Skills
        "cloud computing",
        "devops",
        "ci/cd",
        "containerization",
        "infrastructure as code",
        "version control",
        "cloud architecture",
        "cloud migration",
        "cloud security",
        "container orchestration",
        "configuration management",
        "automation",
        "monitoring",
        "logging",
        "deployment",
        "scaling",
        "load balancing",
        "high availability",
        "disaster recovery",
        "service mesh",
        "site reliability engineering",
        # Security Skills
        "cybersecurity",
        "network security",
        "application security",
        "security assessment",
        "penetration testing",
        "vulnerability assessment",
        "security architecture",
        "identity management",
        "access control",
        "encryption",
        "security monitoring",
        "incident response",
        # Database Skills
        "database administration",
        "database optimization",
        "data modeling",
        "query optimization",
        "database design",
        "database migration",
        "database security",
        "nosql",
        "rdbms",
        "data replication",
        # Soft Skills
        "problem solving",
        "analytical thinking",
        "team collaboration",
        "project management",
        "agile",
        "scrum",
        "communication",
        "leadership",
        "time management",
        "critical thinking",
        "decision making",
        "stakeholder management",
        "presentation",
        "documentation",
        "mentoring",
        "team building",
        "negotiation",
        # Testing Skills
        "software testing",
        "unit testing",
        "integration testing",
        "system testing",
        "performance testing",
        "security testing",
        "automated testing",
        "manual testing",
        "test planning",
        "quality assurance",
        "regression testing",
        "load testing",
        # Architecture Skills
        "system architecture",
        "solution architecture",
        "enterprise architecture",
        "technical architecture",
        "distributed systems",
        "scalable systems",
        "high performance computing",
        "fault tolerance",
        "reliability",
        # Mobile Skills
        "mobile development",
        "ios development",
        "android development",
        "cross platform development",
        "mobile ui design",
        "responsive design",
        "mobile testing",
        "app optimization",
        "mobile security",
    },
    "TECHNOLOGY": {
        # Programming Languages
        "python",
        "java",
        "javascript",
        "typescript",
        "c++",
        "c#",
        "go",
        "rust",
        "swift",
        "kotlin",
        "php",
        "ruby",
        "scala",
        "perl",
        "r",
        "matlab",
        "bash",
        "powershell",
        "sql",
        "dart",
        "lua",
        # Frontend Technologies
        "react",
        "angular",
        "vue",
        "svelte",
        "jquery",
        "bootstrap",
        "tailwind",
        "material ui",
        "webpack",
        "babel",
        "sass",
        "less",
        "html5",
        "css3",
        "redux",
        "next.js",
        "nuxt.js",
        "gatsby",
        # Backend Technologies
        "django",
        "flask",
        "fastapi",
        "spring",
        "spring boot",
        "laravel",
        "express",
        "nest.js",
        "ruby on rails",
        "asp.net",
        "node.js",
        "deno",
        "graphql",
        "rest",
        "grpc",
        "soap",
        # Databases
        "mysql",
        "postgresql",
        "mongodb",
        "redis",
        "elasticsearch",
        "oracle",
        "sql server",
        "cassandra",
        "dynamodb",
        "neo4j",
        "couchdb",
        "mariadb",
        "sqlite",
        "influxdb",
        "timescaledb",
        # Cloud Platforms & Services
        "aws",
        "azure",
        "gcp",
        "alibaba cloud",
        "oracle cloud",
        "digital ocean",
        "heroku",
        "kubernetes",
        "docker",
        "openshift",
        "terraform",
        "cloudformation",
        "ansible",
        "puppet",
        "chef",
        # DevOps Tools
        "git",
        "github",
        "gitlab",
        "bitbucket",
        "jenkins",
        "travis ci",
        "circle ci",
        "github actions",
        "argocd",
        "prometheus",
        "grafana",
        "elk stack",
        "datadog",
        "new relic",
        "splunk",
        "nagios",
        # AI/ML Technologies
        "tensorflow",
        "pytorch",
        "keras",
        "scikit-learn",
        "pandas",
        "numpy",
        "opencv",
        "spacy",
        "nltk",
        "hugging face",
        "mlflow",
        "kubeflow",
        "airflow",
        "spark",
        "hadoop",
        # Testing Tools
        "junit",
        "pytest",
        "selenium",
        "cypress",
        "jest",
        "mocha",
        "karma",
        "postman",
        "jmeter",
        "gatling",
        "cucumber",
        "testng",
        "robot framework",
        "appium",
        # Security Tools
        "wireshark",
        "nmap",
        "metasploit",
        "burp suite",
        "owasp zap",
        "snort",
        "nessus",
        "kali linux",
        "hashicorp vault",
        "sonarqube",
        # Development Tools
        "vscode",
        "intellij",
        "eclipse",
        "sublime text",
        "vim",
        "docker desktop",
        "postman",
        "insomnia",
        "dbeaver",
        "sourcetree",
        # Project Management Tools
        "jira",
        "confluence",
        "trello",
        "asana",
        "notion",
        "slack",
        "microsoft teams",
        "zoom",
        "linear",
        "clickup",
    },
}


class JobDatasetProcessor:
    def __init__(self, raw_data_path: str, processed_data_path: str):
        """
        ตัวประมวลผลข้อมูลงาน สำหรับการเตรียม NER dataset

        Args:
            raw_data_path: path ไปยังไฟล์ data-jobs.csv
            processed_data_path: path สำหรับบันทึกไฟล์ที่ประมวลผลแล้ว
        """
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path

        # สร้าง directory สำหรับเก็บข้อมูลที่ประมวลผลแล้ว
        os.makedirs(processed_data_path, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """โหลดข้อมูลดิบและเลือกคอลัมน์ที่จำเป็น"""
        df = pd.read_csv(self.raw_data_path)

        needed_columns = [
            "Topic",
            "Position",
            "Qualification",
            "Qualification2",
            "Benefits",
            "Province",
            "workTime",
            "Salary",
        ]

        return df[needed_columns]

    def filter_it_jobs(self, df: pd.DataFrame) -> pd.DataFrame:
        """กรองเฉพาะงานด้าน IT/Computer"""
        it_keywords = {
            "software",
            "developer",
            "engineer",
            "programmer",
            "data",
            "system",
            "network",
            "cloud",
            "devops",
            "it",
            "computer",
            "web",
            "application",
            "security",
            "database",
            "analytics",
            "machine learning",
            "ai",
        }

        def is_it_job(row):
            text = " ".join(
                [
                    str(row["Position"]).lower(),
                    str(row["Topic"]).lower(),
                    str(row["Qualification"]).lower(),
                    str(row["Qualification2"]).lower(),
                ]
            )
            return any(keyword in text for keyword in it_keywords)

        filtered_df = df[df.apply(is_it_job, axis=1)]

        print(f"จำนวนงานทั้งหมด: {len(df)}")
        print(f"จำนวนงาน IT: {len(filtered_df)}")
        print(f"สัดส่วนงาน IT: {len(filtered_df)/len(df)*100:.2f}%")

        return filtered_df

    def clean_text(self, text: str) -> str:
        """ทำความสะอาดข้อความ"""
        if pd.isna(text):
            return ""

        # ลบ HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # แทนที่อักขระพิเศษด้วยช่องว่าง
        text = re.sub(r"[^\w\s\.,\-\(\)]", " ", text)

        # แยกคำที่ติดกัน
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

        # แยกตัวเลขกับตัวอักษร
        text = re.sub(r"(\d+)([a-zA-Z])", r"\1 \2", text)
        text = re.sub(r"([a-zA-Z])(\d+)", r"\1 \2", text)

        # ลบช่องว่างซ้ำ
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def tokenize(self, text: str) -> List[str]:
        """แบ่งข้อความเป็น tokens"""
        # ทำความสะอาดข้อความก่อน
        text = self.clean_text(text)

        # แบ่งประโยคด้วยจุด
        sentences = text.split(".")

        # แบ่งคำในแต่ละประโยค
        tokens = []
        for sentence in sentences:
            if sentence.strip():
                # แบ่งด้วยช่องว่างและเครื่องหมายวรรคตอน
                sentence_tokens = re.findall(r"\b\w+\b", sentence.lower())
                tokens.extend(sentence_tokens)

        return tokens

    def create_ner_labels(self, tokens: List[str]) -> List[int]:
        """สร้าง NER labels สำหรับ tokens"""
        labels = ["O"] * len(tokens)

        # ตรวจสอบ multi-word entities ก่อน
        i = 0
        while i < len(tokens):
            found_entity = False
            # ลองรวมคำหลายๆ คำเข้าด้วยกัน
            for j in range(min(5, len(tokens) - i), 0, -1):  # ลองรวมสูงสุด 5 คำ
                phrase = " ".join(tokens[i : i + j]).lower()

                # ตรวจสอบว่าเป็น entity หรือไม่
                for entity_type, keywords in KEYWORDS.items():
                    if phrase in keywords:
                        # ทำ labeling แบบ BIO
                        labels[i] = f"B-{entity_type}"
                        for k in range(i + 1, i + j):
                            labels[k] = f"I-{entity_type}"
                        i += j
                        found_entity = True
                        break
                if found_entity:
                    break

            if not found_entity:
                # ถ้าไม่พบ multi-word entity ให้ตรวจสอบคำเดี่ยว
                token = tokens[i].lower()
                for entity_type, keywords in KEYWORDS.items():
                    if token in keywords:
                        labels[i] = f"B-{entity_type}"
                        break
                i += 1

        return [label2id[label] for label in labels]

    def process_data(self) -> Dict:
        """ประมวลผลข้อมูลทั้งหมด"""
        # โหลดข้อมูล
        df = self.load_data()

        # กรองงาน IT
        df = self.filter_it_jobs(df)

        # ทำความสะอาดข้อความ
        df["Qualification"] = df["Qualification"].apply(self.clean_text)
        df["Qualification2"] = df["Qualification2"].apply(self.clean_text)

        # สร้าง tokens
        df["Qualification_tokens"] = df["Qualification"].apply(self.tokenize)
        df["Qualification2_tokens"] = df["Qualification2"].apply(self.tokenize)

        # กรองแถวที่ไม่มีข้อมูล
        df = df[
            (df["Qualification_tokens"].apply(len) > 0)
            | (df["Qualification2_tokens"].apply(len) > 0)
        ]

        # สร้าง tokens และ labels
        tokens_list = []
        labels_list = []

        for q1, q2 in zip(df["Qualification_tokens"], df["Qualification2_tokens"]):
            if len(q1) > 0:
                tokens_list.append(q1)
                labels_list.append(self.create_ner_labels(q1))
            if len(q2) > 0:
                tokens_list.append(q2)
                labels_list.append(self.create_ner_labels(q2))

        # สร้าง Dataset object
        from datasets import Dataset, DatasetDict

        # แบ่ง train/val/test (80:10:10)
        n = len(tokens_list)
        train_idx = int(0.8 * n)
        val_idx = int(0.9 * n)

        # สร้าง datasets แยกตาม split
        dataset_dict = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "tokens": tokens_list[:train_idx],
                        "ner_tags": labels_list[:train_idx],
                    }
                ),
                "validation": Dataset.from_dict(
                    {
                        "tokens": tokens_list[train_idx:val_idx],
                        "ner_tags": labels_list[train_idx:val_idx],
                    }
                ),
                "test": Dataset.from_dict(
                    {"tokens": tokens_list[val_idx:], "ner_tags": labels_list[val_idx:]}
                ),
            }
        )

        # แสดงสถิติ
        self._print_statistics(df)
        self._print_label_statistics(labels_list)
        self._print_split_statistics(dataset_dict)

        return dataset_dict

    def _print_statistics(self, df: pd.DataFrame):
        """แสดงสถิติของข้อมูล"""
        print("\nสถิติข้อมูล:")
        print(f"จำนวนประกาศงานทั้งหมด: {len(df)}")

        # ความยาวเฉลี่ยของ tokens
        qual1_lengths = df["Qualification_tokens"].apply(len)
        qual2_lengths = df["Qualification2_tokens"].apply(len)
        all_lengths = pd.concat([qual1_lengths, qual2_lengths])

        print(f"ความยาวเฉ���ี่ยของ tokens: {all_lengths.mean():.1f}")
        print(f"ความยาวต่ำสุด: {all_lengths.min()}")
        print(f"ความยาวสูงสุด: {all_lengths.max()}")

        # แสดงการกระจายตัวของตำแหน่งงาน
        print("\nTop 10 ตำแหน่งงาน:")
        position_counts = df["Position"].value_counts().head(10)
        for pos, count in position_counts.items():
            print(f"{pos}: {count}")

    def _print_label_statistics(self, labels_list: List[List[int]]):
        """แสดงสถิติของ labels"""
        print("\nสถิติ Labels:")
        label_counts = {label: 0 for label in id2label.values()}

        for labels in labels_list:
            for label_id in labels:
                label = id2label[label_id]
                label_counts[label] += 1

        total_tokens = sum(label_counts.values())
        for label, count in label_counts.items():
            print(f"{label}: {count} tokens ({count/total_tokens*100:.1f}%)")

    def _print_split_statistics(self, dataset_dict):
        """แสดงสถิติของแต่ละ split"""
        print("\nสถิติการแบ่ง dataset:")
        for split, dataset in dataset_dict.items():
            n_sequences = len(dataset)
            n_tokens = sum(len(tokens) for tokens in dataset["tokens"])
            n_entities = sum(
                sum(1 for tag in tags if tag != label2id["O"])
                for tags in dataset["ner_tags"]
            )

            print(f"\n{split.upper()}:")
            print(f"จำนวน sequences: {n_sequences}")
            print(f"จำนวน tokens ทั้งหมด: {n_tokens}")
            print(f"จำนวน entities: {n_entities}")
            print(f"เฉลี่ย entities ต่อ sequence: {n_entities/n_sequences:.2f}")

    def save_processed_data(self, dataset_dict):
        """บันทึกข้อมูลที่ประมวลผลแล้ว"""
        # บันทึกในรูปแบบ dataset
        output_path = os.path.join(self.processed_data_path, "processed_jobs")
        dataset_dict.save_to_disk(output_path)
        print(f"\nบันทึกข้อมูล dataset ที่: {output_path}")

        # บันทึกเป็น CSV แยกตาม split
        for split, dataset in dataset_dict.items():
            csv_path = os.path.join(
                self.processed_data_path, f"processed_jobs_{split}.csv"
            )

            # แปลง dataset เป็น DataFrame
            df = pd.DataFrame(
                {"tokens": dataset["tokens"], "ner_tags": dataset["ner_tags"]}
            )

            # แปลง labels กลับเป็นชื่อ (O, B-SKILL, etc.)
            df["ner_tags"] = df["ner_tags"].apply(
                lambda x: [id2label[label_id] for label_id in x]
            )

            df.to_csv(csv_path, index=False)
            print(f"บันทึกข้อมูล {split} CSV ที่: {csv_path}")


if __name__ == "__main__":
    RAW_DATA_DIR = "../../data/raw/data-jobs.csv"
    PROCESSED_DATA_DIR = "../../data/processed"

    # สร้าง processor
    processor = JobDatasetProcessor(RAW_DATA_DIR, PROCESSED_DATA_DIR)

    # ประมวลผลข้อมูล
    processed_df = processor.process_data()

    # บันทึกข้อมูล
    processor.save_processed_data(processed_df)
