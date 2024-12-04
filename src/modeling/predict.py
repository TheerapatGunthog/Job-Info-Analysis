import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict, Any

MODEL_DIR = "./models"


class JobNERPredictor:
    """
    คลาสสำหรับทำนาย Named Entities จากข้อความคุณสมบัติงาน
    """

    def __init__(self, model_path: str):
        """
        กำหนดค่าเริ่มต้นสำหรับ predictor

        Args:
            model_path: path ไปยังโฟลเดอร์ที่เก็บโมเดล
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # โหลด tokenizer และ model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # โหลด label mapping
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        ทำนาย entities จากข้อความ

        Args:
            text: ข้อความที่ต้องการวิเคราะห์

        Returns:
            List ของ entities ที่พบ แต่ละ entity ประกอบด้วย:
            - label: ประเภทของ entity
            - text: ข้อความของ entity
            - confidence: ค่าความมั่นใจในการทำนาย
        """
        # แยกข้อความเป็น tokens
        words = text.split()

        # Tokenize
        inputs = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )

        # ย้ายข้อมูลไปยัง device ที่ใช้
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # ทำนาย
        with torch.no_grad():
            outputs = self.model(**inputs)

        # คำนวณ probabilities และ predictions
        probabilities = outputs.logits.softmax(dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)

        # แปลงผลลัพธ์
        results = []
        current_entity = None
        numeric_prefix = None

        # ปรับปรุงการตัดสินใจ
        confidence_threshold = 0.4  # ลด threshold ลง
        min_token_length = 2
        stop_words = {
            "and",
            "or",
            "in",
            "with",
            "for",
            "the",
            "a",
            "an",
            "to",
            "of",
            "on",
            "at",
            "by",
            "up",
            "as",
            "is",
            "are",
            "was",
            "were",
            "be",
            "this",
            "that",
            "these",
            "those",
            "from",
            "needed",
            "required",
            "looking",
            "seeking",
            "wanted",
            "preferred",
            "must",
            "should",
        }

        for idx, (word, pred_idx) in enumerate(zip(words, predictions[0])):
            pred_label = self.id2label[pred_idx.item()]
            confidence = probabilities[0, idx, pred_idx].item()

            # ข้ามคำที่ไม่ต้องการ
            if (
                len(word) < min_token_length
                or word.lower() in stop_words
                or word.lower().endswith((",", ".", ":", ";"))
            ):
                continue

            # ใช้ context window ในการพิจารณา
            context_confidence = confidence
            if idx > 0:
                prev_confidence = probabilities[0, idx - 1, pred_idx].item()
                context_confidence = max(context_confidence, prev_confidence)
            if idx < len(words) - 1:
                next_confidence = probabilities[0, idx + 1, pred_idx].item()
                context_confidence = max(context_confidence, next_confidence)

            if context_confidence < confidence_threshold:
                continue

            if word.isdigit() and idx < len(words) - 1:
                numeric_prefix = word
                continue

            if pred_label.startswith("B-"):
                if current_entity:
                    if current_entity["confidence"] >= confidence_threshold:
                        results.append(current_entity)

                entity_text = f"{numeric_prefix} {word}" if numeric_prefix else word
                current_entity = {
                    "text": entity_text,
                    "label": pred_label[2:],
                    "confidence": confidence,
                    "start_idx": idx - (1 if numeric_prefix else 0),
                }
                numeric_prefix = None

            elif (
                pred_label.startswith("I-")
                and current_entity
                and pred_label[2:] == current_entity["label"]
            ):
                current_entity["text"] = f"{current_entity['text']} {word}"
                # ใช้ค่าความมั่นใจสูงสุด
                current_entity["confidence"] = max(
                    current_entity["confidence"], confidence
                )

        # เพิ่มการตรวจสอบ entity สุดท้าย
        if current_entity and current_entity["confidence"] >= confidence_threshold:
            results.append(current_entity)

        # กรอง entities ที่ซ้ำซ้อน
        filtered_results = []
        seen_texts = set()
        for entity in results:
            normalized_text = entity["text"].lower()
            if normalized_text not in seen_texts:
                seen_texts.add(normalized_text)
                filtered_results.append(entity)

        return filtered_results

    def format_results(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """
        จัดรูปแบบผลลัพธ์ให้อ่านง่าย

        Args:
            text: ข้อความต้นฉบับ
            entities: ผลลัพธ์จากการทำนาย

        Returns:
            ข้อความที่จัดรูปแบบแล้ว
        """
        formatted_output = [f"ข้อความ: {text}\n"]
        formatted_output.append("\nEntities ที่พบ:")

        if not entities:
            formatted_output.append("ไม่พบ entities")
            return "\n".join(formatted_output)

        # จัดกลุ่ม entities ตามประเภท
        entities_by_type = {}
        for entity in entities:
            entity_type = entity["label"]
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)

        # แสดงผลแยกตามประเภท
        type_order = ["ROLE", "PROGRAMMING", "TECHNOLOGY", "SKILL", "EXPERIENCE"]
        type_names = {
            "ROLE": "ตำแหน่งงาน",
            "PROGRAMMING": "ภาษาโปรแกรมมิ่ง",
            "TECHNOLOGY": "เทคโนโลยี",
            "SKILL": "ทักษะ",
            "EXPERIENCE": "ประสบการณ์",
        }

        # เพิ่มการจัดเรียงตาม confidence
        for entity_type in type_order:
            if entity_type in entities_by_type:
                formatted_output.append(f"\n{type_names[entity_type]}:")
                sorted_entities = sorted(
                    entities_by_type[entity_type],
                    key=lambda x: x["confidence"],
                    reverse=True,
                )
                for entity in sorted_entities:
                    formatted_output.append(
                        f"  - {entity['text']} (ความมั่นใจ: {entity['confidence']:.3f})"
                    )

        return "\n".join(formatted_output)


def main():
    # กำหนด path ไปยังโมเดล
    model_path = MODEL_DIR

    # สร้าง predictor
    predictor = JobNERPredictor(model_path)

    # ตัวอย่างข้อความทดสอบ
    test_texts = [
        "Looking for a Senior Software Engineer with 5 years experience in Python and React",
        "Backend Developer needed with skills in Java Spring Boot and MySQL",
        "Data Scientist position requiring expertise in Machine Learning and Python programming",
        "DevOps Engineer with knowledge in Docker, Kubernetes, and AWS",
        "Full Stack Developer proficient in JavaScript, Node.js, and React",
    ]

    print("\nเริ่มการทำนาย:")
    print("=" * 80)

    for text in test_texts:
        # ทำนายและแสดงผล
        entities = predictor.predict(text)
        formatted_output = predictor.format_results(text, entities)
        print(f"\n{formatted_output}")
        print("=" * 80)


if __name__ == "__main__":
    main()
