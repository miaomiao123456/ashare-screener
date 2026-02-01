"""
data_fetcher.py - AKShare数据获取与缓存层
"""
import os
import pickle
import time
import hashlib
import logging
import socket
import akshare as ak
import pandas as pd
from typing import Optional, Any
from functools import wraps

# 设置socket默认超时
socket.setdefaulttimeout(30)

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)


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


def safe_akshare_call(func, *args, max_retries=3, delay=2, **kwargs):
    """安全的AKShare API调用，自动重试"""
    for attempt in range(max_retries):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"AKShare API {func.__name__} failed after {max_retries} attempts: {e}")
                raise
            logger.warning(f"AKShare API {func.__name__} attempt {attempt + 1} failed: {e}, retrying in {delay}s...")
            time.sleep(delay)
    return None


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
        from datetime import datetime
        mtime = os.path.getmtime(path)
        return datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
    except:
        return None


# ============ 基础数据 ============

@retry_on_error(max_retries=2, delay=2)
def get_stock_list() -> pd.DataFrame:
    """获取全部A股代码和名称，失败时使用备用文件"""
    # 优先使用缓存
    cached = cache_get("stock_list", 24)
    if cached is not None:
        logger.info(f"使用缓存的股票列表: {len(cached)} 只股票")
        return cached

    # 尝试从AKShare获取
    logger.info("从AKShare获取股票列表...")
    try:
        df = safe_akshare_call(ak.stock_info_a_code_name, max_retries=2, delay=2)
        if df is not None and not df.empty:
            cache_set("stock_list", df)
            logger.info(f"成功获取 {len(df)} 只股票")
            return df
    except Exception as e:
        logger.warning(f"AKShare获取失败: {e}, 尝试使用备用文件...")

    # 尝试使用备用文件
    backup_path = os.path.join(os.path.dirname(__file__), 'stock_list_backup.pkl')
    if os.path.exists(backup_path):
        try:
            with open(backup_path, 'rb') as f:
                df = pickle.load(f)
            logger.info(f"使用备用文件: {len(df)} 只股票")
            # 更新缓存
            cache_set("stock_list", df)
            return df
        except Exception as e:
            logger.error(f"读取备用文件失败: {e}")

    # 最后尝试一次
    logger.warning("最后一次尝试从AKShare获取...")
    df = safe_akshare_call(ak.stock_info_a_code_name, max_retries=1, delay=5)
    if df is not None and not df.empty:
        cache_set("stock_list", df)
        return df

    raise Exception("无法获取股票列表：AKShare连接失败且无备用文件")


def get_stock_info(code: str) -> pd.DataFrame:
    """获取个股基本信息（行业、市值等）"""
    key = f"info_{code}"
    cached = cache_get(key, 6)
    if cached is not None:
        return cached
    try:
        df = safe_akshare_call(ak.stock_individual_info_em, symbol=code)
        cache_set(key, df)
        return df
    except Exception as e:
        logger.warning(f"get_stock_info({code}): {e}")
        return pd.DataFrame()


# ============ 财务报表 ============

def get_profit_statement(code: str) -> pd.DataFrame:
    """获取利润表（新浪）"""
    key = f"profit_{code}"
    cached = cache_get(key, 12)
    if cached is not None:
        return cached
    stock = f"sh{code}" if code.startswith('6') else f"sz{code}"
    try:
        df = safe_akshare_call(ak.stock_financial_report_sina, stock=stock, symbol='利润表')
        cache_set(key, df)
        return df
    except Exception as e:
        logger.warning(f"get_profit_statement({code}): {e}")
        return pd.DataFrame()


def get_balance_sheet(code: str) -> pd.DataFrame:
    """获取资产负债表（新浪）"""
    key = f"balance_{code}"
    cached = cache_get(key, 12)
    if cached is not None:
        return cached
    stock = f"sh{code}" if code.startswith('6') else f"sz{code}"
    try:
        df = safe_akshare_call(ak.stock_financial_report_sina, stock=stock, symbol='资产负债表')
        cache_set(key, df)
        return df
    except Exception as e:
        logger.warning(f"get_balance_sheet({code}): {e}")
        return pd.DataFrame()


def get_cashflow_statement(code: str) -> pd.DataFrame:
    """获取现金流量表"""
    key = f"cashflow_{code}"
    cached = cache_get(key, 12)
    if cached is not None:
        return cached
    stock = f"sh{code}" if code.startswith('6') else f"sz{code}"
    try:
        df = safe_akshare_call(ak.stock_financial_report_sina, stock=stock, symbol='现金流量表')
        cache_set(key, df)
        return df
    except Exception as e:
        logger.warning(f"get_cashflow_statement({code}): {e}")
        return pd.DataFrame()


# ============ 分红数据 ============

def get_dividend_history(code: str) -> pd.DataFrame:
    """获取个股分红历史"""
    key = f"dividend_{code}"
    cached = cache_get(key, 24)
    if cached is not None:
        return cached
    try:
        df = safe_akshare_call(ak.stock_fhps_detail_em, symbol=code)
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
        df = safe_akshare_call(ak.stock_qbzf_em)
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
        df = safe_akshare_call(ak.bond_cov_stock_issue_cninfo)
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
        df = safe_akshare_call(ak.stock_hold_control_cninfo, symbol="全部")
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
        df = safe_akshare_call(ak.stock_main_stock_holder, stock=code)
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
        df = safe_akshare_call(ak.stock_repurchase_em)
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
        df = safe_akshare_call(ak.stock_gpzy_pledge_ratio_detail_em)
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
        df = safe_akshare_call(
            ak.stock_zh_a_hist,
            symbol=code, period="daily",
            start_date=start_date, end_date=end_date,
            adjust="qfq"
        )
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
