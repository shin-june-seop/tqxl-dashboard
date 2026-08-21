# Leveraged Strategy Dashboard V1.1

V1에 매매일지와 현재 홀딩스를 추가했습니다.

## 기능
- TQQQ/QQQ, SOXL/SOXX 2주 전략 판정
- BUY/SELL 매매일지
- 수량/매수가/환율/수수료/메모 기록
- 매매일지에서 현재 홀딩스 자동 계산
- 평균매수가/현재가/평가액/미실현손익/수익률
- 전략 목표비중 vs 실제비중 비교

## Supabase
`supabase_strategy_schema.sql`을 Supabase SQL Editor에서 한 번 실행합니다.
Streamlit Cloud Secrets:
SUPABASE_URL=...
SUPABASE_KEY=...

기존 V15 프로젝트와 별도 테이블 `strategy_trades`만 사용하므로 기존 `transactions` 테이블과 충돌하지 않습니다.
