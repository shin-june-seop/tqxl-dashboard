# Leveraged Strategy Dashboard V1

TQQQ/QQQ와 SOXL/SOXX의 2주 판정 전략을 자동 계산하는 별도 Streamlit 앱입니다.

## 핵심 규칙
- 일봉 데이터로 200일 이동평균 계산
- 2주마다 상태 판정
- 강세장: 200MA 위 2회 연속
- 초기 하락: 첫 200MA 이탈
- 진성 하락: 200MA 아래 3회 연속 또는 이격도 기준
- 4단계: 200MA 대비 -25% 이하
- 상승 복귀: 재돌파 후 2주차
- 급락 시 월중 비상 대응
- 레버리지 ETF 자체 과열 시 버블 보험 경고

## Streamlit Cloud
GitHub 저장소에 `app.py`, `requirements.txt`를 올리고 Main file path를 `app.py`로 지정하면 됩니다.
