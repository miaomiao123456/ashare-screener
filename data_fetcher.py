"""
data_fetcher.py - Tushare数据获取与缓存层
"""
import os
import pickle
import time
import hashlib
import logging
import tushare as ts
import pandas as pd
from typing import Optional, Any
from functools import wraps
from datetime import datetime, timedelta
from threading import Lock

logger = logging.getLogger(__name__)

# Tushare Token
TUSHARE_TOKEN = '308865f85885fd56a54f5dc6a0f58d7ec03c57f5365263ea3e9f6409'
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# API限速：Tushare每分钟200次调用
_api_call_times = []
_api_lock = Lock()
API_RATE_LIMIT = 180  # 每分钟180次，留20次余量
API_RATE_WINDOW = 60  # 60秒窗口


def _rate_limit():
    """API限速控制"""
    global _api_call_times
    with _api_lock:
        now = time.time()
        # 清理60秒前的调用记录
        _api_call_times = [t for t in _api_call_times if now - t < API_RATE_WINDOW]

        # 如果达到限制，等待
        if len(_api_call_times) >= API_RATE_LIMIT:
            oldest = _api_call_times[0]
            wait_time = API_RATE_WINDOW - (now - oldest) + 0.1
            if wait_time > 0:
                logger.warning(f"API频率限制，等待 {wait_time:.1f}秒...")
                time.sleep(wait_time)
                _api_call_times = []

        _api_call_times.append(now)


def retry_on_error(max_retries=3, delay=2):
    """重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts: {e}")
                        raise
                    logger.warning(f"{func.__name__} attempt {attempt + 1} failed: {e}, retrying...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


def safe_tushare_call(func, max_retries=3, delay=1, **kwargs):
    """安全的Tushare API调用，自动重试和限速"""
    for attempt in range(max_retries):
        try:
            _rate_limit()  # API限速
            result = func(**kwargs)
            if result is not None and not result.empty:
                return result
            elif attempt < max_retries - 1:
                time.sleep(delay)
                continue
            return pd.DataFrame()
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Tushare API failed after {max_retries} attempts: {e}")
                return pd.DataFrame()
            logger.warning(f"Tushare API attempt {attempt + 1} failed: {e}, retrying in {delay}s...")
            time.sleep(delay)
    return pd.DataFrame()


def _cache_path(key: str) -> str:
    safe = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{safe}.pkl")


def cache_get(key: str, max_age_hours: float = 24) -> Optional[Any]:
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    if (time.time() - os.path.getmtime(path)) > max_age_hours * 3600:
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def cache_set(key: str, data: Any):
    try:
        with open(_cache_path(key), 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        logger.warning(f"Cache write failed: {e}")


def get_cache_update_time(key: str) -> Optional[str]:
    """获取缓存数据的更新时间"""
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    try:
        mtime = os.path.getmtime(path)
        return datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
    except:
        return None


# ============ 基础数据 ============

@retry_on_error(max_retries=2, delay=2)
def get_stock_list() -> pd.DataFrame:
    """获取全部A股代码和名称"""
    cached = cache_get("stock_list", 24)
    if cached is not None:
        logger.info(f"使用缓存的股票列表: {len(cached)} 只股票")
        return cached

    logger.info("从Tushare获取股票列表...")
    try:
        # 获取A股列表（L=上市 D=退市 P=暂停上市）
        df = safe_tushare_call(pro.stock_basic, exchange='', list_status='L',
                               fields='ts_code,symbol,name,area,industry,market')
        if not df.empty:
            # 转换为AKShare格式：code, name
            df['code'] = df['symbol']
            df = df[['code', 'name']]
            cache_set("stock_list", df)
            logger.info(f"成功获取 {len(df)} 只股票")
            return df
    except Exception as e:
        logger.warning(f"Tushare获取失败: {e}")

    # 尝试使用备用文件
    backup_path = os.path.join(os.path.dirname(__file__), 'stock_list_backup.pkl')
    if os.path.exists(backup_path):
        try:
            with open(backup_path, 'rb') as f:
                df = pickle.load(f)
            logger.info(f"使用备用文件: {len(df)} 只股票")
            cache_set("stock_list", df)
            return df
        except Exception as e:
            logger.error(f"读取备用文件失败: {e}")

    raise Exception("无法获取股票列表：Tushare连接失败且无备用文件")


def get_stock_info(code: str) -> pd.DataFrame:
    """获取个股基本信息（市值、价格等）

    返回格式：item, value
    """
    key = f"info_{code}"
    cached = cache_get(key, 12)  # 增加缓存到12小时
    if cached is not None:
        return cached

    try:
        ts_code = _convert_code_to_ts(code)

        # 获取基本信息
        basic_info = safe_tushare_call(pro.stock_basic, ts_code=ts_code)
        # 获取最新行情
        daily_info = safe_tushare_call(pro.daily_basic, ts_code=ts_code,
                                       trade_date=_get_latest_trade_date(),
                                       fields='ts_code,close,turnover_rate,pe_ttm,pb,total_mv,circ_mv')

        # 组装为AKShare格式
        result = []
        if not basic_info.empty:
            row = basic_info.iloc[0]
            result.append({'item': '股票代码', 'value': code})
            result.append({'item': '股票名称', 'value': row.get('name', '')})
            result.append({'item': '行业', 'value': row.get('industry', '')})
            result.append({'item': '地域', 'value': row.get('area', '')})

        if not daily_info.empty:
            row = daily_info.iloc[0]
            result.append({'item': '最新价', 'value': row.get('close', 0)})
            result.append({'item': '市盈率', 'value': row.get('pe_ttm', 0)})
            result.append({'item': '市净率', 'value': row.get('pb', 0)})
            result.append({'item': '总市值', 'value': row.get('total_mv', 0)})
            result.append({'item': '流通市值', 'value': row.get('circ_mv', 0)})
            result.append({'item': '换手率', 'value': row.get('turnover_rate', 0)})

        df = pd.DataFrame(result)
        cache_set(key, df)
        return df
    except Exception as e:
        logger.warning(f"get_stock_info({code}): {e}")
        return pd.DataFrame()


def get_financial_indicator(code: str) -> pd.DataFrame:
    """获取财务指标（包含股息率、ROE等）- 用于快速筛选

    返回最近8个季度的财务指标
    """
    key = f"fina_indicator_{code}"
    cached = cache_get(key, 24)  # 缓存24小时
    if cached is not None:
        return cached

    try:
        ts_code = _convert_code_to_ts(code)
        # 获取最近8个季度的财务指标
        df = safe_tushare_call(pro.fina_indicator, ts_code=ts_code,
                              fields='ts_code,end_date,roe,roa,gross_profit_margin,netprofit_margin,debt_to_assets,current_ratio,quick_ratio')

        if not df.empty:
            df = df.sort_values('end_date', ascending=False).head(8)
            cache_set(key, df)
        return df
    except Exception as e:
        logger.debug(f"get_financial_indicator({code}): {e}")
        return pd.DataFrame()


# ============ 业绩预告 ============

def get_forecast(code: str) -> pd.DataFrame:
    """获取业绩预告"""
    key = f"forecast_{code}"
    cached = cache_get(key, 6)  # 缓存6小时，业绩预告更新较频繁
    if cached is not None:
        return cached

    try:
        ts_code = _convert_code_to_ts(code)
        # 获取最近2年的业绩预告
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y%m%d')

        df = safe_tushare_call(pro.forecast, ts_code=ts_code,
                              start_date=start_date, end_date=end_date,
                              fields='ts_code,ann_date,end_date,type,p_change_min,p_change_max,net_profit_min,net_profit_max,summary')

        if not df.empty:
            df['公告日期'] = df['ann_date']
            df['报告期'] = df['end_date']
            df['业绩变动类型'] = df['type']
            df['净利润变动幅度最小值'] = df['p_change_min']
            df['净利润变动幅度最大值'] = df['p_change_max']
            df['预告净利润最小值'] = df['net_profit_min']
            df['预告净利润最大值'] = df['net_profit_max']
            df['业绩预告摘要'] = df['summary']
            df = df.sort_values('ann_date', ascending=False)
            cache_set(key, df)
        return df
    except Exception as e:
        logger.warning(f"get_forecast({code}): {e}")
        return pd.DataFrame()


# ============ 财务报表 ============

def get_profit_statement(code: str) -> pd.DataFrame:
    """获取利润表"""
    key = f"profit_{code}"
    cached = cache_get(key, 48)  # 增加缓存到48小时
    if cached is not None:
        return cached

    try:
        ts_code = _convert_code_to_ts(code)
        # 获取近3年利润表（按季度）
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=1095)).strftime('%Y%m%d')

        df = safe_tushare_call(pro.income, ts_code=ts_code,
                              start_date=start_date, end_date=end_date,
                              fields='ts_code,end_date,total_revenue,revenue,n_income,n_income_attr_p')

        if not df.empty:
            # 转换为AKShare格式
            df['报告日'] = df['end_date']
            df['营业总收入'] = df['total_revenue'].fillna(df['revenue'])
            df['净利润'] = df['n_income_attr_p'].fillna(df['n_income'])
            df = df.sort_values('end_date', ascending=False)
            cache_set(key, df)
        return df
    except Exception as e:
        logger.warning(f"get_profit_statement({code}): {e}")
        return pd.DataFrame()


def get_balance_sheet(code: str) -> pd.DataFrame:
    """获取资产负债表"""
    key = f"balance_{code}"
    cached = cache_get(key, 48)  # 增加缓存到48小时
    if cached is not None:
        return cached

    try:
        ts_code = _convert_code_to_ts(code)
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=1095)).strftime('%Y%m%d')

        df = safe_tushare_call(pro.balancesheet, ts_code=ts_code,
                              start_date=start_date, end_date=end_date,
                              fields='ts_code,end_date,money_cap,st_borr,lt_borr,bond_payable')

        if not df.empty:
            df['报告日'] = df['end_date']
            df['货币资金'] = df['money_cap']
            df['短期借款'] = df['st_borr']
            df['长期借款'] = df['lt_borr']
            df['应付债券'] = df['bond_payable']
            df = df.sort_values('end_date', ascending=False)
            cache_set(key, df)
        return df
    except Exception as e:
        logger.warning(f"get_balance_sheet({code}): {e}")
        return pd.DataFrame()


def get_cashflow_statement(code: str) -> pd.DataFrame:
    """获取现金流量表"""
    key = f"cashflow_{code}"
    cached = cache_get(key, 48)  # 增加缓存到48小时
    if cached is not None:
        return cached

    try:
        ts_code = _convert_code_to_ts(code)
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=1095)).strftime('%Y%m%d')

        df = safe_tushare_call(pro.cashflow, ts_code=ts_code,
                              start_date=start_date, end_date=end_date,
                              fields='ts_code,end_date,n_cashflow_act,c_inf_fr_invest_a,c_inf_fr_fin_a,c_paid_invest')

        if not df.empty:
            df['报告日'] = df['end_date']
            df['经营活动现金流'] = df['n_cashflow_act']
            df['投资活动现金流'] = df['c_inf_fr_invest_a']
            df['筹资活动现金流'] = df['c_inf_fr_fin_a']
            df['投资支付的现金'] = df['c_paid_invest']
            df = df.sort_values('end_date', ascending=False)
            cache_set(key, df)
        return df
    except Exception as e:
        logger.warning(f"get_cashflow_statement({code}): {e}")
        return pd.DataFrame()


# ============ 分红数据 ============

def get_dividend_history(code: str) -> pd.DataFrame:
    """获取个股分红历史"""
    key = f"dividend_{code}"
    cached = cache_get(key, 48)  # 增加缓存到48小时
    if cached is not None:
        return cached

    try:
        ts_code = _convert_code_to_ts(code)
        # 获取近10年分红
        df = safe_tushare_call(pro.dividend, ts_code=ts_code,
                              fields='ts_code,ann_date,end_date,div_proc,stk_div,stk_bo_rate,cash_div,cash_div_tax')

        if not df.empty:
            # 中文字段映射
            df['报告期'] = df['end_date']
            df['公告日期'] = df['ann_date']
            df['实施进度'] = df['div_proc']
            df['每10股送股'] = df['stk_div'].fillna(0)
            df['每10股转增'] = df['stk_bo_rate'].fillna(0)
            df['每10股派息(元)'] = df['cash_div'].fillna(0)
            df['每10股派息-税后(元)'] = df['cash_div_tax'].fillna(0)
            # 保留原字段用于兼容性
            df['现金分红-每股派息'] = df['cash_div_tax'].fillna(0) / 10  # Tushare单位是每10股，转为每股
            df['现金分红-现金分红比例'] = df['cash_div'].fillna(0)
            df = df.sort_values('end_date', ascending=False)
            cache_set(key, df)
        return df
    except Exception as e:
        logger.warning(f"get_dividend_history({code}): {e}")
        return pd.DataFrame()


# ============ 增发/发债 ============

def get_additional_issuance() -> pd.DataFrame:
    """获取全部增发数据"""
    cached = cache_get("additional_issuance", 24)
    if cached is not None:
        return cached

    try:
        # 获取近5年增发数据
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=1825)).strftime('%Y%m%d')

        df = safe_tushare_call(pro.share_float, start_date=start_date, end_date=end_date,
                              fields='ts_code,ann_date,float_date,float_share,float_ratio,holder_name')

        if not df.empty:
            # 筛选增发相关记录（排除首发、配股等）
            df = df[df['holder_name'].notna()]
            df['股票代码'] = df['ts_code'].apply(lambda x: x.split('.')[0] if pd.notna(x) else '')
            df['公告日期'] = df['ann_date']
            df = df.sort_values('ann_date', ascending=False)
            cache_set("additional_issuance", df)
        return df
    except Exception as e:
        logger.warning(f"get_additional_issuance: {e}")
        return pd.DataFrame()


def get_convertible_bonds() -> pd.DataFrame:
    """获取可转债发行数据"""
    cached = cache_get("conv_bonds", 24)
    if cached is not None:
        return cached

    try:
        # 获取可转债发行数据
        df = safe_tushare_call(pro.cb_issue,
                              fields='ts_code,bond_short_name,stk_code,ann_date,issue_date,issue_size')

        if not df.empty:
            df['股票代码'] = df['stk_code']
            df['公告日期'] = df['ann_date']
            df['发行日期'] = df['issue_date']
            df = df.sort_values('ann_date', ascending=False)
            cache_set("conv_bonds", df)
        return df
    except Exception as e:
        logger.warning(f"get_convertible_bonds: {e}")
        return pd.DataFrame()


# ============ 股东/控制人 ============

def get_controller_info() -> pd.DataFrame:
    """获取全部上市公司实际控制人信息"""
    cached = cache_get("controller_info", 24)
    if cached is not None:
        return cached

    try:
        # Tushare使用stk_holdertrade获取控制权变更信息
        # 这里使用股东信息作为替代
        df = safe_tushare_call(pro.stk_holdertrade,
                              fields='ts_code,ann_date,holder_name,holder_type')

        if not df.empty:
            df['证券代码'] = df['ts_code'].apply(lambda x: x.split('.')[0] if pd.notna(x) else '')
            df['实际控制人名称'] = df['holder_name']
            df['控制类型'] = df['holder_type']
            cache_set("controller_info", df)
        return df
    except Exception as e:
        logger.warning(f"get_controller_info: {e}")
        return pd.DataFrame()


def get_shareholder_info(code: str) -> pd.DataFrame:
    """获取十大股东信息"""
    key = f"shareholders_{code}"
    cached = cache_get(key, 24)
    if cached is not None:
        return cached

    try:
        ts_code = _convert_code_to_ts(code)
        # 获取最新的十大股东
        df = safe_tushare_call(pro.top10_holders, ts_code=ts_code,
                              fields='ts_code,end_date,holder_name,hold_amount,hold_ratio')

        if not df.empty:
            # 中文字段映射
            df['报告期'] = df['end_date']
            df['股东名称'] = df['holder_name']
            df['持股数量'] = df['hold_amount']
            df['持股比例(%)'] = df['hold_ratio']
            df = df.sort_values('end_date', ascending=False)
            cache_set(key, df)
        return df
    except Exception as e:
        logger.warning(f"get_shareholder_info({code}): {e}")
        return pd.DataFrame()


# ============ 回购数据 ============

def get_buyback_data() -> pd.DataFrame:
    """获取全部股票回购数据"""
    cached = cache_get("buyback_data", 12)
    if cached is not None:
        return cached

    try:
        # 获取回购数据
        df = safe_tushare_call(pro.repurchase,
                              fields='ts_code,ann_date,end_date,proc,exp_date,vol,amount,high_limit,low_limit')

        if not df.empty:
            df['股票代码'] = df['ts_code'].apply(lambda x: x.split('.')[0] if pd.notna(x) else '')
            df['实施进度'] = df['proc']
            df = df.sort_values('ann_date', ascending=False)
            cache_set("buyback_data", df)
        return df
    except Exception as e:
        logger.warning(f"get_buyback_data: {e}")
        return pd.DataFrame()


# ============ 质押数据 ============

def get_pledge_data() -> pd.DataFrame:
    """获取全部股权质押数据"""
    cached = cache_get("pledge_data", 24)
    if cached is not None:
        return cached

    try:
        # 获取质押统计数据
        df = safe_tushare_call(pro.pledge_stat,
                              fields='ts_code,end_date,pledge_count,unrest_pledge,rest_pledge,total_share,pledge_ratio')

        if not df.empty:
            df['股票代码'] = df['ts_code'].apply(lambda x: x.split('.')[0] if pd.notna(x) else '')
            df['质押比例'] = df['pledge_ratio']
            df = df.sort_values('end_date', ascending=False)
            cache_set("pledge_data", df)
        return df
    except Exception as e:
        logger.warning(f"get_pledge_data: {e}")
        return pd.DataFrame()


# ============ K线行情 ============

def get_kline(code: str, start_date: str = "20200101", end_date: str = "20261231") -> pd.DataFrame:
    """获取K线行情数据"""
    key = f"kline_{code}_{start_date}_{end_date}"
    cached = cache_get(key, 4)
    if cached is not None:
        return cached

    try:
        ts_code = _convert_code_to_ts(code)
        # 使用pro_bar获取前复权数据
        df = ts.pro_bar(ts_code=ts_code, adj='qfq',
                       start_date=start_date, end_date=end_date,
                       factors=['tor', 'vr'])

        if df is not None and not df.empty:
            # 转换为AKShare格式
            df['日期'] = df['trade_date']
            df['开盘'] = df['open']
            df['收盘'] = df['close']
            df['最高'] = df['high']
            df['最低'] = df['low']
            df['成交量'] = df['vol']
            df = df.sort_values('trade_date')
            cache_set(key, df)
            return df
    except Exception as e:
        logger.warning(f"get_kline({code}): {e}")

    return pd.DataFrame()


# ============ 板块数据 ============

def get_stock_sector(code: str) -> dict:
    """获取个股所属板块"""
    info = get_stock_info(code)
    result = {"industry": "", "sector": ""}
    if not info.empty:
        for _, row in info.iterrows():
            item = str(row.get('item', ''))
            value = str(row.get('value', ''))
            if '行业' in item:
                result['industry'] = value
    return result


# ============ 辅助函数 ============

def _convert_code_to_ts(code: str) -> str:
    """将6位代码转换为Tushare格式 (如 000001.SZ)"""
    if code.startswith('6'):
        return f"{code}.SH"
    elif code.startswith(('0', '3')):
        return f"{code}.SZ"
    elif code.startswith('4') or code.startswith('8'):
        return f"{code}.BJ"
    return f"{code}.SH"


def _get_latest_trade_date() -> str:
    """获取最近的交易日"""
    # 简单实现：返回最近3天内的日期（考虑周末）
    for i in range(5):
        date = (datetime.now() - timedelta(days=i)).strftime('%Y%m%d')
        # 这里可以优化为查询交易日历，暂时简化处理
        return date
    return datetime.now().strftime('%Y%m%d')
