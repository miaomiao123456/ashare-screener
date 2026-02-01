# -*- coding: utf-8 -*-
"""
聚宽平台回测策略 - A股8大条件筛选
复制此代码到聚宽(JoinQuant)平台运行回测
"""

from jqdata import *
from jqfactor import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def initialize(context):
    """初始化"""
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    set_slippage(FixedSlippage(0.02))
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001,
                              open_commission=0.0003, close_commission=0.0003,
                              min_commission=5), type='stock')

    # 每月第一个交易日调仓
    run_monthly(rebalance, 1, time='open')

    g.hold_num = 10  # 最大持仓数

def rebalance(context):
    """月度调仓"""
    target_stocks = screen_stocks(context)

    # 卖出不在目标列表的股票
    for stock in context.portfolio.positions:
        if stock not in target_stocks:
            order_target(stock, 0)

    # 等权买入目标股票
    if len(target_stocks) > 0:
        cash_per_stock = context.portfolio.available_cash / len(target_stocks)
        for stock in target_stocks[:g.hold_num]:
            order_value(stock, cash_per_stock)

# ============================================================
#                    8大条件筛选主函数
# ============================================================

def screen_stocks(context):
    """执行8大条件筛选"""
    today = context.current_dt.date()

    # 获取所有A股（排除ST、停牌、退市）
    all_stocks = get_all_securities(types=['stock'], date=today)
    stocks = [s for s in all_stocks.index if not s.startswith('8')]  # 排除北交所
    stocks = filter_st_paused(stocks, today)

    log.info(f"初始股票池: {len(stocks)}")

    # 条件1: 连续3年年报营收净利增长
    stocks = filter_3year_growth(stocks, today)
    log.info(f"条件1(3年增长)后: {len(stocks)}")

    # 条件2: 最新季报同比环比增长
    stocks = filter_quarterly_growth(stocks, today)
    log.info(f"条件2(季报增长)后: {len(stocks)}")

    # 条件3: 连续5年现金分红，无增发发债
    stocks = filter_dividend_no_financing(stocks, today)
    log.info(f"条件3(分红无融资)后: {len(stocks)}")

    # 条件4: 股息率 > 4%
    stocks = filter_dividend_yield(stocks, today)
    log.info(f"条件4(股息率>4%)后: {len(stocks)}")

    # 条件5: 前两大股东是央国企
    stocks = filter_soe_top2(stocks, today)
    log.info(f"条件5(央国企)后: {len(stocks)}")

    # 条件6: 有回购行为
    stocks = filter_buyback(stocks, today)
    log.info(f"条件6(回购)后: {len(stocks)}")

    # 条件7: 实控人未变更
    stocks = filter_controller_stable(stocks, today)
    log.info(f"条件7(实控人稳定)后: {len(stocks)}")

    # 条件8: 货币资金 > 有息负债
    stocks = filter_cash_gt_debt(stocks, today)
    log.info(f"条件8(现金>负债)后: {len(stocks)}")

    return stocks

# ============================================================
#                    辅助函数
# ============================================================

def filter_st_paused(stocks, date):
    """过滤ST和停牌股票"""
    # 过滤ST
    st_stocks = get_extras('is_st', stocks, end_date=date, count=1).iloc[0]
    stocks = [s for s in stocks if not st_stocks.get(s, True)]

    # 过滤停牌
    paused = get_price(stocks, end_date=date, count=1, fields=['paused'])['paused'].iloc[0]
    stocks = [s for s in stocks if not paused.get(s, True)]

    return stocks

# ============================================================
#                    条件1: 连续3年年报营收净利增长
# ============================================================

def filter_3year_growth(stocks, date):
    """
    连续3年年报营收净利润增长
    判断逻辑：
    - 获取最近4个年报（考虑年报4月底前披露）
    - 要求：Year1 > Year2 > Year3 > Year4（营收和净利润都满足）
    - 基数必须为正（排除亏损转盈利的情况）
    """
    passed = []

    # 确定最近4个可用年报期
    year = date.year
    month = date.month

    # 年报在次年4月30日前披露完毕
    if month >= 5:
        latest_annual = year - 1
    else:
        latest_annual = year - 2

    annual_dates = [f"{y}-12-31" for y in range(latest_annual, latest_annual-4, -1)]

    for stock in stocks:
        try:
            revenues = []
            profits = []
            valid = True

            for report_date in annual_dates:
                q = query(
                    income.statDate,
                    income.total_operating_revenue,
                    income.net_profit
                ).filter(
                    income.code == stock,
                    income.statDate == report_date
                )
                df = get_fundamentals(q)

                if df.empty:
                    valid = False
                    break

                rev = df['total_operating_revenue'].iloc[0]
                profit = df['net_profit'].iloc[0]

                if pd.isna(rev) or pd.isna(profit):
                    valid = False
                    break

                revenues.append(rev)
                profits.append(profit)

            if not valid or len(revenues) < 4:
                continue

            # 检查连续3年增长
            growth_ok = True
            for i in range(3):
                if revenues[i+1] <= 0 or profits[i+1] <= 0:
                    growth_ok = False
                    break
                if revenues[i] <= revenues[i+1] or profits[i] <= profits[i+1]:
                    growth_ok = False
                    break

            if growth_ok:
                passed.append(stock)

        except Exception as e:
            continue

    return passed

# ============================================================
#                    条件2: 季报同比环比增长
# ============================================================

def filter_quarterly_growth(stocks, date):
    """
    最新季报营收净利同比、环比双增长
    判断逻辑：
    - 同比：本季 vs 去年同季（如2024Q3 vs 2023Q3）
    - 环比：本季 vs 上季（如2024Q3 vs 2024Q2，Q1则对比上年Q4）
    - 要求：营收和净利润同时满足同比+环比双增长
    - 基数必须为正
    """
    passed = []

    for stock in stocks:
        try:
            q = query(
                income.statDate,
                income.total_operating_revenue,
                income.net_profit
            ).filter(
                income.code == stock
            ).order_by(
                income.statDate.desc()
            ).limit(8)

            df = get_fundamentals(q, statDate=date)
            if df.empty or len(df) < 5:
                continue

            df = df.sort_values('statDate', ascending=False).reset_index(drop=True)

            latest = df.iloc[0]
            latest_date = str(latest['statDate'])
            latest_quarter = latest_date[5:7]
            latest_year = latest_date[:4]

            latest_rev = latest['total_operating_revenue']
            latest_profit = latest['net_profit']

            if pd.isna(latest_rev) or pd.isna(latest_profit):
                continue

            # 上一季度（环比）
            prev_q = df.iloc[1] if len(df) > 1 else None
            if prev_q is None:
                continue

            prev_q_rev = prev_q['total_operating_revenue']
            prev_q_profit = prev_q['net_profit']

            # 去年同期（同比）
            target_yoy_date = f"{int(latest_year)-1}-{latest_quarter}"
            yoy_row = None
            for i in range(1, len(df)):
                if str(df.iloc[i]['statDate']).startswith(target_yoy_date[:7]):
                    yoy_row = df.iloc[i]
                    break

            if yoy_row is None:
                continue

            yoy_rev = yoy_row['total_operating_revenue']
            yoy_profit = yoy_row['net_profit']

            # 同比增长检查
            if yoy_rev <= 0 or yoy_profit <= 0:
                continue
            if latest_rev <= yoy_rev or latest_profit <= yoy_profit:
                continue

            # 环比增长检查
            if prev_q_rev <= 0 or prev_q_profit <= 0:
                continue
            if latest_rev <= prev_q_rev or latest_profit <= prev_q_profit:
                continue

            passed.append(stock)

        except Exception as e:
            continue

    return passed

# ============================================================
#                    条件3: 连续5年分红，无增发发债
# ============================================================

def filter_dividend_no_financing(stocks, date):
    """
    连续5年现金分红 + 近5年无增发或发债
    判断逻辑：
    - 分红：检查过去5个完整财年每年都有现金分红记录（用bonus_year字段）
    - 增发：5年内无定向增发、公开增发记录
    - 发债：5年内无可转债发行记录
    """
    passed = []

    year = date.year
    month = date.month

    # 分红通常在次年实施
    if month >= 7:
        check_years = list(range(year-1, year-6, -1))
    else:
        check_years = list(range(year-2, year-7, -1))

    five_years_ago = date - timedelta(days=5*365)

    for stock in stocks:
        try:
            # ===== 检查连续5年分红 =====
            q = query(
                finance.STK_XR_XD.company_id,
                finance.STK_XR_XD.bonus_year,
                finance.STK_XR_XD.cash_before_tax,
                finance.STK_XR_XD.a_registration_date
            ).filter(
                finance.STK_XR_XD.code == stock,
                finance.STK_XR_XD.cash_before_tax > 0
            )
            div_df = finance.run_query(q)

            if div_df.empty:
                continue

            div_years = set(div_df['bonus_year'].dropna().astype(int))
            if not set(check_years).issubset(div_years):
                continue

            # ===== 检查无增发 =====
            q_capital = query(
                finance.STK_CAPITAL_CHANGE.code,
                finance.STK_CAPITAL_CHANGE.change_date,
                finance.STK_CAPITAL_CHANGE.change_reason_id
            ).filter(
                finance.STK_CAPITAL_CHANGE.code == stock,
                finance.STK_CAPITAL_CHANGE.change_date >= five_years_ago,
                finance.STK_CAPITAL_CHANGE.change_reason_id.in_([303, 304, 305, 306])
            )
            capital_df = finance.run_query(q_capital)

            if not capital_df.empty:
                continue

            # ===== 检查无发债 =====
            q_bond = query(
                bond.CONBOND_BASIC_INFO.code,
                bond.CONBOND_BASIC_INFO.list_date
            ).filter(
                bond.CONBOND_BASIC_INFO.company_code == stock,
                bond.CONBOND_BASIC_INFO.list_date >= five_years_ago
            )
            bond_df = bond.run_query(q_bond)

            if not bond_df.empty:
                continue

            passed.append(stock)

        except Exception as e:
            continue

    return passed

# ============================================================
#                    条件4: 股息率 > 4%
# ============================================================

def filter_dividend_yield(stocks, date):
    """
    股息率 > 4%
    判断逻辑：
    - 股息率 = 过去12个月每股现金分红(税前) / 当前股价 × 100%
    - 使用股权登记日在过去12个月内的分红记录
    - 当前股价使用最新收盘价
    """
    passed = []

    prices = get_price(stocks, end_date=date, count=1, fields=['close'])['close']
    if prices.empty:
        return passed

    current_prices = prices.iloc[0]
    one_year_ago = date - timedelta(days=365)

    for stock in stocks:
        try:
            price = current_prices.get(stock)
            if pd.isna(price) or price <= 0:
                continue

            q = query(
                finance.STK_XR_XD.code,
                finance.STK_XR_XD.cash_before_tax,
                finance.STK_XR_XD.a_registration_date
            ).filter(
                finance.STK_XR_XD.code == stock,
                finance.STK_XR_XD.a_registration_date >= one_year_ago,
                finance.STK_XR_XD.a_registration_date <= date,
                finance.STK_XR_XD.cash_before_tax > 0
            )
            div_df = finance.run_query(q)

            if div_df.empty:
                continue

            total_div = div_df['cash_before_tax'].sum()
            dividend_yield = (total_div / price) * 100

            if dividend_yield >= 4.0:
                passed.append(stock)

        except Exception as e:
            continue

    return passed

# ============================================================
#                    条件5: 前两大股东是央国企
# ============================================================

SOE_KEYWORDS = [
    '国有', '国资', '财政局', '财政厅', '国投', '中央', '省人民政府',
    '市人民政府', '国家', '央企', '国企', '人民政府', '管理委员会',
    '国有资产', '财政部', '国务院', '中国人民', '省国资', '市国资',
    '区国资', '县国资', '经济开发区', '高新区管委会', '国有独资',
    '中央汇金', '社保基金', '梧桐树投资', '中国证券金融', '中国烟草',
    '中国石油', '中国石化', '中国移动', '中国电信', '中国联通',
    '国家电网', '中国铁路', '中国邮政', '中国航空', '中国船舶'
]

def is_soe(name):
    """判断是否央国企"""
    if not name or pd.isna(name):
        return False
    name = str(name)
    return any(kw in name for kw in SOE_KEYWORDS)

def filter_soe_top2(stocks, date):
    """
    前两大股东都是央国企
    判断逻辑：
    - 获取最新一期十大股东数据
    - 检查第1大和第2大股东名称
    - 名称包含央国企关键词则判定为央国企
    - 要求前两大股东同时满足
    """
    passed = []

    for stock in stocks:
        try:
            q = query(
                finance.STK_SHAREHOLDER_TOP10.code,
                finance.STK_SHAREHOLDER_TOP10.shareholder_name,
                finance.STK_SHAREHOLDER_TOP10.shareholder_rank,
                finance.STK_SHAREHOLDER_TOP10.end_date
            ).filter(
                finance.STK_SHAREHOLDER_TOP10.code == stock
            ).order_by(
                finance.STK_SHAREHOLDER_TOP10.end_date.desc(),
                finance.STK_SHAREHOLDER_TOP10.shareholder_rank.asc()
            ).limit(20)

            sh_df = finance.run_query(q)

            if sh_df.empty:
                continue

            latest_date = sh_df['end_date'].max()
            latest_sh = sh_df[sh_df['end_date'] == latest_date].sort_values('shareholder_rank')

            if len(latest_sh) < 2:
                continue

            top1_name = latest_sh.iloc[0]['shareholder_name']
            top2_name = latest_sh.iloc[1]['shareholder_name']

            if is_soe(top1_name) and is_soe(top2_name):
                passed.append(stock)

        except Exception as e:
            continue

    return passed

# ============================================================
#                    条件6: 有回购行为
# ============================================================

def filter_buyback(stocks, date):
    """
    公司有回购股票行为
    判断逻辑：
    - 检查近2年内是否有股票回购记录
    - 回购状态为：已完成、实施中、董事会预案、股东大会通过
    - 包括公司回购和大股东增持
    """
    passed = []
    two_years_ago = date - timedelta(days=2*365)

    for stock in stocks:
        try:
            q = query(
                finance.STK_SHARES_REPURCHASE.code,
                finance.STK_SHARES_REPURCHASE.end_date,
                finance.STK_SHARES_REPURCHASE.repurchase_state
            ).filter(
                finance.STK_SHARES_REPURCHASE.code == stock,
                finance.STK_SHARES_REPURCHASE.end_date >= two_years_ago
            )
            buyback_df = finance.run_query(q)

            if buyback_df.empty:
                continue

            valid_states = ['已完成', '实施中', '董事会预案', '股东大会通过', '已实施']
            has_buyback = buyback_df['repurchase_state'].isin(valid_states).any()

            if has_buyback:
                passed.append(stock)

        except Exception as e:
            continue

    return passed

# ============================================================
#                    条件7: 实控人未变更
# ============================================================

def filter_controller_stable(stocks, date):
    """
    实际控制人近3年未变更
    判断逻辑：
    - 获取近3年的实际控制人变更记录
    - 如果实控人名称唯一，则视为稳定
    - 无记录的也视为稳定（未发生变更）
    """
    passed = []
    three_years_ago = date - timedelta(days=3*365)

    for stock in stocks:
        try:
            q = query(
                finance.STK_CONTROLLER_INFO.code,
                finance.STK_CONTROLLER_INFO.controller_name,
                finance.STK_CONTROLLER_INFO.end_date
            ).filter(
                finance.STK_CONTROLLER_INFO.code == stock,
                finance.STK_CONTROLLER_INFO.end_date >= three_years_ago
            ).order_by(
                finance.STK_CONTROLLER_INFO.end_date.desc()
            )
            ctrl_df = finance.run_query(q)

            if ctrl_df.empty:
                passed.append(stock)
                continue

            controllers = ctrl_df['controller_name'].dropna().unique()

            if len(controllers) <= 1:
                passed.append(stock)

        except Exception as e:
            passed.append(stock)
            continue

    return passed

# ============================================================
#                    条件8: 货币资金 > 有息负债
# ============================================================

def filter_cash_gt_debt(stocks, date):
    """
    货币资金 > 有息负债总额
    判断逻辑：
    - 有息负债 = 短期借款 + 长期借款 + 应付债券 + 一年内到期非流动负债
    - 要求：货币资金 > 有息负债
    - 同时检查：股权质押比例 <= 30%
    """
    passed = []

    for stock in stocks:
        try:
            q = query(
                balance.code,
                balance.statDate,
                balance.cash_equivalents,
                balance.shortterm_loan,
                balance.longterm_loan,
                balance.bonds_payable,
                balance.non_current_liability_in_one_year
            ).filter(
                balance.code == stock
            ).order_by(
                balance.statDate.desc()
            ).limit(1)

            df = get_fundamentals(q, statDate=date)

            if df.empty:
                continue

            row = df.iloc[0]

            cash = row['cash_equivalents'] or 0
            short_loan = row['shortterm_loan'] or 0
            long_loan = row['longterm_loan'] or 0
            bonds = row['bonds_payable'] or 0
            non_current_1y = row['non_current_liability_in_one_year'] or 0

            total_interest_debt = short_loan + long_loan + bonds + non_current_1y

            if cash <= total_interest_debt:
                continue

            # 检查质押比例
            q_pledge = query(
                finance.STK_HOLDER_PLEDGE.code,
                finance.STK_HOLDER_PLEDGE.pledge_ratio
            ).filter(
                finance.STK_HOLDER_PLEDGE.code == stock
            ).order_by(
                finance.STK_HOLDER_PLEDGE.end_date.desc()
            ).limit(1)

            pledge_df = finance.run_query(q_pledge)

            if not pledge_df.empty:
                pledge_ratio = pledge_df.iloc[0]['pledge_ratio']
                if pledge_ratio and pledge_ratio > 30:
                    continue

            passed.append(stock)

        except Exception as e:
            continue

    return passed
