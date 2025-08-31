# 명령어 모음

```bash

# 애플리케이션 실행
lsof -i :8110 | grep LISTEN | awk '{print $2}' | xargs kill -9 && sleep 2 && cd /@@@@/deotis_qa_chatbot && .venv/bin/python src/api/main.py

# 가상환경
source /@@@/deotis_qa_chatbot/.venv/bin/activate

source venv/bin/activate

-------

# 인덱싱

## 기존 데이터 유지하며 파일 타입만 새롭게 추가
python src/api/main.py --init-index --file-types json
python src/api/main.py --init-index --file-types pdf
python src/api/main.py --init-index --file-types txt
python src/api/main.py --init-index --file-types docx
python src/api/main.py --init-index --file-types all

## 새로운 --only 옵션 (기존 데이터 삭제 후 json만 인덱싱)
python src/api/main.py --init-index --file-types json --only

## 다른 파일 타입들도 동일하게 사용 가능
python src/api/main.py --init-index --file-types pdf --only
python src/api/main.py --init-index --file-types txt --only
python src/api/main.py --init-index --file-types docx --only
python src/api/main.py --init-index --file-types all --only

-------

# docx -> json

## 기본 변환
python src/preprocess/run_docx_to_index.py data/docx

## 분석과 비교를 포함한 변환
python src/preprocess/run_docx_to_index.py data/docx --compare data/index.json --analyze

## 특정 출력 파일로 저장
python src/preprocess/run_docx_to_index.py data/docx -o custom_output.json

-------

# elasticsearch

## 'my-index' 라는 이름의 인덱스를 삭제
curl -X DELETE "http://localhost:9200/unified_rag" -v

## 인덱스 삭제 확인
curl -s http://localhost:9200/_cat/indices?v

## docker 재기동
docker-compose down && docker-compose up -d
docker compose stop elasticsearch

```
# 프로젝트 흐름도

<br/>

<img width="5688" height="1406" alt="이미지추출용 excalidraw" src="https://github.com/user-attachments/assets/25116f7c-f964-487f-b742-167091173035" />

<img width="11683" height="1585" alt="이미지추출용 excalidraw" src="https://github.com/user-attachments/assets/b1995084-7d34-4487-9c0b-32cf22d7c8e8" />

<img width="7947" height="2388" alt="이미지추출용 excalidraw" src="https://github.com/user-attachments/assets/012cb1e5-c4f1-4eb2-9d6c-74a7eb8f6bb3" />




