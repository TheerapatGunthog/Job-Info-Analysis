import pandas as pd
import json
from typing import List, Dict
import os
from collections import Counter

PROCESSED_DATA_DIR = "../../data/processed"


class EntityValidator:
    def __init__(self, processed_data_path: str):
        """
        ตัวตรวจสอบความถูกต้องของ entities

        Args:
            processed_data_path: path ไปยังโฟลเดอร์ที่เก็บข้อมูลที่ประมวลผลแล้ว
        """
        self.processed_data_path = processed_data_path

    def load_data(self, split: str = "train") -> pd.DataFrame:
        """โหลดข้อมูลจาก CSV"""
        csv_path = os.path.join(self.processed_data_path, f"processed_jobs_{split}.csv")
        return pd.read_csv(csv_path)

    def validate_bio_format(self, tags: List[str]) -> List[str]:
        """ตรวจสอบความถูกต้องของ BIO format"""
        errors = []
        for i, tag in enumerate(tags):
            # ข้าม O tag
            if tag == "O":
                continue

            # ตรวจสอบ I- tag ที่ไม่มี B- นำหน้า
            if tag.startswith("I-"):
                if i == 0 or not tags[i - 1].startswith(("B-", "I-")):
                    errors.append(f"พบ I- tag ที่ไม่มี B- นำหน้า ที่ตำแหน่ง {i}: {tag}")
                elif not tags[i - 1].endswith(
                    tag[2:]
                ):  # ตรวจสอบว่าเป็น entity type เดียวกัน
                    errors.append(
                        f"พบ I- tag ที่ entity type ไม่ตรงกับ tag ก่อนหน้า ที่ตำแหน่ง {i}: {tag}"
                    )

        return errors

    def analyze_entities(self, df: pd.DataFrame) -> Dict:
        """วิเคราะห์ entities ในข้อมูล"""
        # แปลง string representation เป็น list ถ้าจำเป็น
        if isinstance(df["tokens"].iloc[0], str):
            df["tokens"] = df["tokens"].apply(eval)
            df["ner_tags"] = df["ner_tags"].apply(eval)

        # เก็บสถิติ
        stats = {
            "total_sequences": len(df),
            "total_tokens": sum(len(tokens) for tokens in df["tokens"]),
            "entity_counts": Counter(),
            "entity_examples": {},
            "bio_errors": [],
        }

        # วิเคราะห์แ่ละ sequence
        for tokens, tags in zip(df["tokens"], df["ner_tags"]):
            # ตรวจสอบ BIO format
            errors = self.validate_bio_format(tags)
            stats["bio_errors"].extend(errors)

            # นับจำนวน entities
            current_entity = None
            current_tokens = []

            for token, tag in zip(tokens, tags):
                if tag.startswith("B-"):
                    # จบ entity เก่า (ถ้ามี)
                    if current_entity:
                        entity_text = " ".join(current_tokens)
                        stats["entity_counts"][current_entity] += 1
                        stats["entity_examples"].setdefault(current_entity, set()).add(
                            entity_text
                        )

                    # เริ่ม entity ใหม่
                    current_entity = tag[2:]  # ตัด "B-" ออก
                    current_tokens = [token]

                elif tag.startswith("I-"):
                    if current_entity:
                        current_tokens.append(token)

                else:  # O tag
                    # จบ entity เก่า (ถ้ามี)
                    if current_entity:
                        entity_text = " ".join(current_tokens)
                        stats["entity_counts"][current_entity] += 1
                        stats["entity_examples"].setdefault(current_entity, set()).add(
                            entity_text
                        )
                        current_entity = None
                        current_tokens = []

            # จบ entity สุดท้าย (ถ้ามี)
            if current_entity:
                entity_text = " ".join(current_tokens)
                stats["entity_counts"][current_entity] += 1
                stats["entity_examples"].setdefault(current_entity, set()).add(
                    entity_text
                )

        return stats

    def print_validation_report(self, stats: Dict, df: pd.DataFrame):
        """พิมพ์รายงานการตรวจสอบ"""
        print("\n=== รายงานการตรวจสอบ Entities ===")
        print(f"\nจำนวน sequences ทั้งหมด: {stats['total_sequences']}")
        print(f"จำนวน tokens ทั้งหมด: {stats['total_tokens']}")

        print("\nจำนวน entities แต่ละประเภท:")
        for entity_type, count in stats["entity_counts"].items():
            print(f"{entity_type}: {count}")

        print("\nตัวอย่าง entities แต่ละประเภท (สูงสุด 5 ตัวอย่าง):")
        for entity_type, examples in stats["entity_examples"].items():
            print(f"\n{entity_type}:")
            for example in list(examples)[:5]:
                print(f"  - {example}")

        if stats["bio_errors"]:
            print("\nข้อผิดพลาดที่พบ:")
            for error in stats["bio_errors"]:
                print(f"- {error}")
        else:
            print("\nไม่พบข้อผิดพลาดในรูปแบบ BIO")

        # แสดงตัวอย่างประโยคที่มี entities
        self.show_entity_examples(df)

    def save_validation_report(self, stats: Dict):
        """บันทึกรายงานการตรวจสอบ"""
        # แปลง set เป็น list เพื่อให้สามารถ serialize เป็น JSON ได้
        stats["entity_examples"] = {
            k: list(v) for k, v in stats["entity_examples"].items()
        }

        # บันทึกเป็น JSON
        output_path = os.path.join(self.processed_data_path, "validation_report.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\nบันทึกรายงานที่: {output_path}")

    def show_entity_examples(self, df: pd.DataFrame, n_examples: int = 5):
        """แสดงตัวอย่างประโยคที่มี entities

        Args:
            df: DataFrame ที่มีคอลัมน์ tokens และ ner_tags
            n_examples: จำนวนตัวอย่างที่ต้องการแสดง
        """
        # ตรวจสอบว่า tokens และ ner_tags เป็น string หรือไม่
        if isinstance(df["tokens"].iloc[0], str):
            df["tokens"] = df["tokens"].apply(eval)
            df["ner_tags"] = df["ner_tags"].apply(eval)

        print("\n=== ตัวอย่างประโยคที่มี Entities ===")

        # เก็บตัวอย่างประโยคที่มี entities แต่ละประเภท
        examples_by_type = {}

        for tokens, tags in zip(df["tokens"], df["ner_tags"]):
            # ตรวจสอบว่ามี entities หรือไม่
            has_entities = any(tag != "O" for tag in tags)
            if not has_entities:
                continue

            # สร้างประโยคพร้อม highlight entities
            formatted_tokens = []
            current_entity = None
            entity_tokens = []

            for token, tag in zip(tokens, tags):
                if tag.startswith("B-"):
                    # จบ entity เก่า (ถ้ามี)
                    if current_entity:
                        entity_text = f"[{current_entity}] {' '.join(entity_tokens)}"
                        formatted_tokens.append(entity_text)

                    # เริ่ม entity ใหม่
                    current_entity = tag[2:]
                    entity_tokens = [token]

                elif tag.startswith("I-"):
                    if current_entity:
                        entity_tokens.append(token)

                else:  # O tag
                    # จบ entity เก่า (ถ้ามี)
                    if current_entity:
                        entity_text = f"[{current_entity}] {' '.join(entity_tokens)}"
                        formatted_tokens.append(entity_text)
                        current_entity = None
                        entity_tokens = []
                    formatted_tokens.append(token)

            # จบ entity สุดท้าย (ถ้ามี)
            if current_entity:
                entity_text = f"[{current_entity}] {' '.join(entity_tokens)}"
                formatted_tokens.append(entity_text)

            # สร้างประโยคที่ format แล้ว
            formatted_text = " ".join(formatted_tokens)

            # เก็บตัวอย่างแยกตาม entity type
            entities_in_text = set(tag[2:] for tag in tags if tag != "O")
            for entity_type in entities_in_text:
                examples_by_type.setdefault(entity_type, []).append(formatted_text)

        # แสดงตัวอย่างแยกตาม entity type
        for entity_type, examples in examples_by_type.items():
            print(f"\n{entity_type}:")
            for i, example in enumerate(examples[:n_examples], 1):
                print(f"{i}. {example}")


def main():
    # สร้าง validator
    validator = EntityValidator(PROCESSED_DATA_DIR)

    # โหลดและตรวจสอบข้อมูลแต่ละ split
    for split in ["train", "validation", "test"]:
        print(f"\nกำลังตรวจสอบข้อมูล {split}...")
        df = validator.load_data(split)
        stats = validator.analyze_entities(df)
        validator.print_validation_report(stats, df)
        validator.save_validation_report(stats)
        validator.show_entity_examples(df)


if __name__ == "__main__":
    main()

    file_path = os.path.join(PROCESSED_DATA_DIR, f"processed_jobs_{"train"}.csv")
    df = pd.read_csv(file_path)
