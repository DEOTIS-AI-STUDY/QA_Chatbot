#!/usr/bin/env python3
"""
DOCX → index.json 변환 실행 스크립트
사용법: python run_docx_to_index.py [옵션] <docx_directory>
"""
import argparse
import os
import sys
import json
from typing import List, Dict, Any

# 모듈 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from improved_index_generator import generate_improved_index
except ImportError:
    # 상대 import 시도
    try:
        from .improved_index_generator import generate_improved_index
    except ImportError:
        print("❌ 오류: improved_index_generator 모듈을 찾을 수 없습니다.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="DOCX 파일들을 index.json 형태로 변환",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  python run_docx_to_index.py data/docx
  python run_docx_to_index.py data/docx -o output.json
  python run_docx_to_index.py data/docx --compare data/index.json
  python run_docx_to_index.py data/docx --analyze
        """
    )
    
    parser.add_argument(
        'directory',
        help='DOCX 파일들이 있는 디렉토리 경로'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='출력 파일 경로 (기본값: <directory>/converted_index.json)',
        default=None
    )
    
    parser.add_argument(
        '--compare',
        help='비교할 기존 index.json 파일 경로',
        default=None
    )
    
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='생성된 데이터의 구조를 분석하여 출력'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='상세한 로그 출력'
    )
    
    args = parser.parse_args()
    
    # 입력 검증
    if not os.path.isdir(args.directory):
        print(f"❌ 오류: 디렉토리를 찾을 수 없습니다: {args.directory}")
        sys.exit(1)
    
    # 출력 파일 경로 설정
    if not args.output:
        # data/json 디렉토리가 있으면 그곳에, 없으면 입력 디렉토리에 생성
        json_dir = "data/json"
        if os.path.exists(json_dir) and os.path.isdir(json_dir):
            args.output = os.path.join(json_dir, "converted_index.json")
        else:
            # data/json 디렉토리가 없으면 생성
            try:
                os.makedirs(json_dir, exist_ok=True)
                args.output = os.path.join(json_dir, "converted_index.json")
                print(f"📁 data/json 디렉토리를 생성했습니다.")
            except Exception as e:
                print(f"⚠️  data/json 디렉토리 생성 실패, 입력 디렉토리에 저장합니다: {e}")
                args.output = os.path.join(args.directory, "converted_index.json")
    
    # DOCX → index.json 변환
    print("🔄 DOCX 파일들을 index.json 형태로 변환 중...")
    try:
        result = generate_improved_index(args.directory, args.output)
        
        if not result:
            print("❌ 변환할 DOCX 파일을 찾을 수 없습니다.")
            sys.exit(1)
        
        print(f"✅ 변환 완료: {args.output}")
        
        # 결과 요약
        total_items = sum(len(item['data']) for item in result)
        print(f"📊 변환 결과: {len(result)}개 파일, 총 {total_items}개 항목")
        
        for item in result:
            print(f"   - {item['filename']}: {len(item['data'])}개 항목")
        
        # 분석 옵션
        if args.analyze:
            analyze_structure(result)
        
        # 비교 옵션
        if args.compare:
            compare_with_original(result, args.compare)
        
    except Exception as e:
        print(f"❌ 변환 중 오류 발생: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def analyze_structure(data: List[Dict[str, Any]]):
    """생성된 데이터의 구조 분석"""
    print("\n📈 구조 분석 결과:")
    
    for file_item in data:
        filename = file_item['filename']
        items = file_item['data']
        
        print(f"\n📄 {filename}:")
        
        # 통계 계산
        title_count = sum(1 for item in items if item.get('title', '').strip())
        heading_count = sum(1 for item in items if item.get('heading', '').strip())
        section_count = sum(1 for item in items if item.get('section', '').strip())
        
        print(f"   📑 전체 항목: {len(items)}개")
        print(f"   📋 Title 항목: {title_count}개")
        print(f"   📝 Heading 항목: {heading_count}개")
        print(f"   📌 Section 항목: {section_count}개")
        
        # 고유 title들 출력
        unique_titles = set(item['title'] for item in items if item.get('title', '').strip())
        if unique_titles:
            print(f"   🏷️  고유 Title들 ({len(unique_titles)}개):")
            for title in sorted(unique_titles)[:5]:  # 처음 5개만
                print(f"      - {title}")
            if len(unique_titles) > 5:
                print(f"      ... 및 {len(unique_titles) - 5}개 더")


def compare_with_original(generated: List[Dict[str, Any]], original_path: str):
    """기존 index.json과 비교"""
    print(f"\n🔍 원본 파일과 비교: {original_path}")
    
    try:
        with open(original_path, 'r', encoding='utf-8') as f:
            original = json.load(f)
        
        print("📊 비교 결과:")
        
        # 파일 개수 비교
        print(f"   파일 개수: 원본 {len(original)}개 vs 생성됨 {len(generated)}개")
        
        # 각 파일별 항목 수 비교
        original_dict = {item['filename']: len(item['data']) for item in original}
        generated_dict = {item['filename']: len(item['data']) for item in generated}
        
        all_files = set(original_dict.keys()) | set(generated_dict.keys())
        
        for filename in sorted(all_files):
            orig_count = original_dict.get(filename, 0)
            gen_count = generated_dict.get(filename, 0)
            
            if orig_count == 0:
                print(f"   + {filename}: 새로 생성됨 ({gen_count}개)")
            elif gen_count == 0:
                print(f"   - {filename}: 누락됨 (원본 {orig_count}개)")
            else:
                diff = gen_count - orig_count
                diff_str = f"(+{diff})" if diff > 0 else f"({diff})" if diff < 0 else "(동일)"
                print(f"   📄 {filename}: 원본 {orig_count}개 → 생성 {gen_count}개 {diff_str}")
        
        # 구조 유사성 체크
        if generated and original:
            check_structure_similarity(generated[0]['data'][:3], original[0]['data'][:3])
        
    except FileNotFoundError:
        print(f"❌ 원본 파일을 찾을 수 없습니다: {original_path}")
    except json.JSONDecodeError:
        print(f"❌ 원본 파일이 유효한 JSON이 아닙니다: {original_path}")


def check_structure_similarity(generated_sample: List[Dict], original_sample: List[Dict]):
    """구조 유사성 확인"""
    print("   🔍 구조 유사성 검사 (첫 3개 항목):")
    
    for i, (gen_item, orig_item) in enumerate(zip(generated_sample, original_sample), 1):
        print(f"      항목 {i}:")
        
        # 필드 존재 여부 비교
        gen_fields = set(gen_item.keys())
        orig_fields = set(orig_item.keys())
        
        if gen_fields == orig_fields:
            print(f"        ✅ 필드 구조 일치: {sorted(gen_fields)}")
        else:
            missing = orig_fields - gen_fields
            extra = gen_fields - orig_fields
            if missing:
                print(f"        ⚠️  누락된 필드: {sorted(missing)}")
            if extra:
                print(f"        ➕ 추가된 필드: {sorted(extra)}")


if __name__ == "__main__":
    main()
