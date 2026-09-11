import argparse
from .market_data import YFinanceProvider, ingest_market_data
from .sample_data import make_sample_data
from .backtest import run_simple_backtest

def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["backtest", "download"])
    p.add_argument("--symbol")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--interval", default="5m")
    args = p.parse_args()
    if args.command == "backtest":
        df = make_sample_data()
        trades = run_simple_backtest(df)
        print(f"Trades generated: {len(trades)}")
        if trades:
            print(f"Raw P&L: {sum(t.pnl for t in trades):.2f}")
            print("NOTE: sample data only; not evidence of strategy profitability.")
    if args.command == "download":
        if not all([args.symbol, args.start, args.end]):
            p.error("download requires --symbol, --start, and --end")
        result = ingest_market_data(
            YFinanceProvider(), args.symbol, args.start, args.end, args.interval
        )
        print(f"Raw data: {result.raw_path}")
        print(f"Validated data: {result.processed_path}")
        print(result.report)

if __name__ == "__main__":
    main()
