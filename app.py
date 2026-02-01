"""
app.py - A股优质股票筛选系统 Flask主应用
"""
import os
import sys
import json
import logging
import threading
from datetime import datetime
from flask import Flask, render_template, jsonify, request

# 确保当前目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data_fetcher as df
from screener import StockScreener

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')

# ============ 全局筛选状态 ============
screening_state = {
    'is_running': False,
    'progress': {'message': '', 'stage': '', 'remaining': 0},
    'results': None,
    'error': None
}


# ============ 页面路由 ============
@app.route('/')
def index():
    return render_template('index.html')


# ============ 筛选 API ============
@app.route('/api/screen/start', methods=['POST'])
def start_screening():
    global screening_state
    if screening_state['is_running']:
        return jsonify({'error': '筛选正在进行中'}), 400

    # 获取用户选择的条件
    data = request.get_json(silent=True) or {}
    selected_criteria = data.get('criteria', list(range(1, 9)))  # 默认全选

    screening_state['is_running'] = True
    screening_state['progress'] = {'message': '初始化...', 'stage': 'init', 'remaining': 0}
    screening_state['results'] = None
    screening_state['error'] = None

    def run():
        global screening_state
        try:
            def progress_cb(info):
                screening_state['progress'] = info

            screener = StockScreener(progress_callback=progress_cb)
            results = screener.screen(selected_criteria=selected_criteria)
            screening_state['results'] = results
        except Exception as e:
            logger.error(f"Screening failed: {e}", exc_info=True)
            screening_state['error'] = str(e)
        finally:
            screening_state['is_running'] = False

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return jsonify({'status': 'started'})


@app.route('/api/screen/progress')
def get_progress():
    return jsonify({
        'is_running': screening_state['is_running'],
        'progress': screening_state['progress'],
        'error': screening_state['error']
    })


@app.route('/api/screen/results')
def get_results():
    if screening_state['results'] is None:
        return jsonify({'error': '暂无筛选结果'}), 404
    return jsonify(screening_state['results'])


# ============ 个股详情 API ============
@app.route('/api/stock/<code>/info')
def stock_info(code):
    """个股基本信息"""
    try:
        info = df.get_stock_info(code)
        if info.empty:
            return jsonify({'error': '未找到'}), 404

        result = {}
        for _, row in info.iterrows():
            result[str(row.get('item', ''))] = str(row.get('value', ''))
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stock/<code>/kline')
def stock_kline(code):
    """K线行情数据"""
    start = request.args.get('start', '20200101')
    end = request.args.get('end', datetime.now().strftime('%Y%m%d'))
    try:
        kdf = df.get_kline(code, start, end)
        if kdf.empty:
            return jsonify({'dates': [], 'ohlc': [], 'volumes': []})

        data = {
            'dates': kdf['日期'].astype(str).tolist(),
            'ohlc': kdf[['开盘', '收盘', '最低', '最高']].values.tolist(),
            'volumes': kdf['成交量'].astype(float).tolist(),
            'closes': kdf['收盘'].astype(float).tolist()
        }
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stock/<code>/financials')
def stock_financials(code):
    """近5年财报数据"""
    try:
        profit = df.get_profit_statement(code)
        balance = df.get_balance_sheet(code)
        cashflow = df.get_cashflow_statement(code)

        def to_annual(frame):
            if frame.empty:
                return []
            annual = frame[frame['报告日'].astype(str).str.endswith('1231')].head(5)
            return annual.fillna(0).to_dict(orient='records')

        return jsonify({
            'profit': to_annual(profit),
            'balance': to_annual(balance),
            'cashflow': to_annual(cashflow)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stock/<code>/shareholders')
def stock_shareholders(code):
    """股东信息"""
    try:
        sh_df = df.get_shareholder_info(code)
        if sh_df.empty:
            return jsonify([])
        # 取最新一期的前10大股东
        if '股东名称' in sh_df.columns:
            return jsonify(sh_df.head(10).fillna('').to_dict(orient='records'))
        return jsonify(sh_df.head(10).fillna('').to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stock/<code>/dividend')
def stock_dividend(code):
    """分红历史"""
    try:
        div_df = df.get_dividend_history(code)
        if div_df.empty:
            return jsonify([])
        return jsonify(div_df.head(10).fillna(0).to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stock/<code>/sector')
def stock_sector(code):
    """所属板块"""
    try:
        sector = df.get_stock_sector(code)
        return jsonify(sector)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
