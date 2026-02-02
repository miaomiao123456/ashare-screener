"""
screener.py - A股8大条件筛选引擎
渐进式过滤：先用批量API淘汰大部分，再逐只精细检查
"""
import logging
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import data_fetcher as df

logger = logging.getLogger(__name__)

# 央国企关键词
SOE_KEYWORDS = [
    '国有', '国资', '财政局', '财政厅', '国投', '中央', '省人民政府',
    '市人民政府', '国家', '央企', '国企', '人民政府', '管理委员会',
    '国有资产', 'SASAC', '财政部', '国务院', '中国人民', '省国资',
    '市国资', '区国资', '县国资', '经济开发区', '高新区管委会'
]


def _is_soe(name: str) -> bool:
    if not name or pd.isna(name):
        return False
    name = str(name)
    return any(kw in name for kw in SOE_KEYWORDS)


def _safe_float(val, default=0.0) -> float:
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


class StockScreener:
    def __init__(self, progress_callback=None):
        self.progress_cb = progress_callback

    def _report(self, msg: str, stage: str = "", remaining: int = 0):
        if self.progress_cb:
            self.progress_cb({
                'message': msg,
                'stage': stage,
                'remaining': remaining
            })
        logger.info(f"[筛选] {msg} | 剩余: {remaining}")

    # ============================
    #  主筛选入口
    # ============================
    def screen(self, selected_criteria=None) -> dict:
        """执行完整筛选流程，返回结果

        Args:
            selected_criteria: 选中的条件ID列表 [1-8]，None表示全选
        """
        if selected_criteria is None:
            selected_criteria = list(range(1, 9))
        selected = set(selected_criteria)
        logger.info(f"[筛选] 实际执行的条件ID: {sorted(selected)}")

        stock_df = df.get_stock_list()
        # 过滤ST、退市、北交所
        all_codes = []
        for _, row in stock_df.iterrows():
            code = str(row['code']).zfill(6)
            name = str(row.get('name', ''))
            if 'ST' in name or '退' in name:
                continue
            # 仅保留沪深主板+创业板+科创板
            if code.startswith(('00', '60', '30', '68')):
                all_codes.append(code)

        total = len(all_codes)
        self._report(f"共获取 {total} 只A股（排除ST/退市）", "init", total)

        # 获取数据日期信息
        from datetime import datetime
        data_dates = {
            'screening_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        # 获取各类数据的更新时间
        stock_list_time = df.get_cache_update_time("stock_list")
        if stock_list_time:
            data_dates['stock_list_update'] = stock_list_time

        # 尝试获取最新财报日期
        try:
            if all_codes:
                sample_code = all_codes[0]
                sample_profit = df.get_profit_statement(sample_code)
                if not sample_profit.empty:
                    latest_report = sample_profit.iloc[0].get('报告日', '')
                    if latest_report:
                        data_dates['latest_financial_report'] = str(latest_report)[:10]

                # 获取最新股价数据更新时间
                price_time = df.get_cache_update_time(f"price_{sample_code}")
                if price_time:
                    data_dates['price_data_update'] = price_time
        except Exception as e:
            logger.debug(f"获取数据日期信息失败: {e}")

        results = {
            'total_initial': total,
            'stages': [],
            'passed': all_codes,
            'stock_names': {},
            'data_dates': data_dates
        }

        # 保存名称映射
        name_map = {}
        for _, row in stock_df.iterrows():
            name_map[str(row['code']).zfill(6)] = str(row.get('name', ''))
        results['stock_names'] = name_map

        # ---- Phase 1: 批量过滤 ----
        # 条件ID映射: 5=央国企, 3=分红+无增发发债, 6=回购
        all_batch_filters = [
            (5, "央国企控股", self._filter_soe),
            (3, "连续5年现金分红", self._filter_dividends),
            (3, "无增发/发债", self._filter_no_issuance),
            (6, "大股东回购", self._filter_buyback),
        ]
        filters = [(name, func) for cid, name, func in all_batch_filters if cid in selected]
        logger.info(f"[批量过滤] 将执行 {len(filters)} 个条件: {[n for n, _ in filters]}")
        for name, func in filters:
            before = len(results['passed'])
            self._report(f"正在检查: {name}", name, before)
            try:
                results['passed'] = func(results['passed'])
            except Exception as e:
                logger.error(f"Filter {name} failed: {e}")
            after = len(results['passed'])
            results['stages'].append({
                'criterion': name,
                'before': before,
                'after': after,
                'eliminated': before - after
            })
            self._report(f"{name}: {before} → {after}", name, after)
            if not results['passed']:
                break

        # ---- Phase 2: 逐只过滤 ----
        # 条件ID映射: 4=股息率, 1=3年增长, 2=季度增长, 7=实控人, 8=现金>负债
        all_individual_filters = [
            (4, "股息率>4%", self._check_dividend_yield),
            (1, "3年年报营收净利增长", self._check_3year_growth),
            (2, "季度同比环比增长", self._check_quarterly_growth),
            (7, "实控人稳定", self._check_controller_stable),
            (8, "现金>负债", self._check_cash_gt_debt),
        ]
        individual_filters = [(name, func) for cid, name, func in all_individual_filters if cid in selected]
        logger.info(f"[逐只过滤] 将执行 {len(individual_filters)} 个条件: {[n for n, _ in individual_filters]}")
        for name, func in individual_filters:
            if not results['passed']:
                break
            before = len(results['passed'])
            self._report(f"逐只检查: {name}", name, before)
            passed = self._run_individual(results['passed'], func, name)
            after = len(passed)
            results['stages'].append({
                'criterion': name,
                'before': before,
                'after': after,
                'eliminated': before - after
            })
            results['passed'] = passed
            self._report(f"{name}: {before} → {after}", name, after)

        results['final_count'] = len(results['passed'])
        results['selected_criteria'] = list(selected)
        self._report(f"筛选完成（{len(selected)}项条件），共 {results['final_count']} 只符合条件", "done", results['final_count'])
        return results

    def _run_individual(self, codes: list, check_func, name: str) -> list:
        """多线程逐只检查"""
        passed = []
        skipped = 0  # 统计因API失败跳过的股票数
        # 增加并发数以提升速度
        max_workers = min(16, len(codes))  # 最多16个线程，或股票数量
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(check_func, c): c for c in codes}
            done = 0
            for future in as_completed(futures):
                code = futures[future]
                done += 1
                if done % 20 == 0 or done == len(codes):  # 每20个更新一次进度
                    self._report(f"{name}: 已检查 {done}/{len(codes)}", name, len(codes) - done)
                try:
                    result = future.result()
                    if result is True:
                        passed.append(code)
                    elif result is None:
                        # None表示API失败，跳过而不过滤
                        skipped += 1
                        passed.append(code)  # 保留该股票，继续后续检查
                    # result is False表示不符合条件，不添加到passed
                except Exception as e:
                    logger.warning(f"Check {name} for {code} failed: {e}")
                    skipped += 1
                    passed.append(code)  # 异常时也保留，避免误杀

        if skipped > 0:
            logger.info(f"[{name}] 因API失败跳过 {skipped} 只股票")
        return passed

    # ============================
    #  Phase 1: 批量过滤函数
    # ============================

    def _filter_soe(self, codes: list) -> list:
        """过滤：前二大股东是央国企"""
        controller_df = df.get_controller_info()
        if controller_df.empty:
            logger.warning("无法获取控制人数据，跳过央国企筛选")
            return codes

        soe_codes = set()
        for _, row in controller_df.iterrows():
            raw_code = str(row.get('证券代码', '')).strip()
            # 可能是 "000001" 或带前缀
            code = raw_code.replace('sh', '').replace('sz', '').zfill(6)
            controller = str(row.get('实际控制人名称', ''))
            ctrl_type = str(row.get('控制类型', ''))
            if _is_soe(controller) or '国有' in ctrl_type:
                soe_codes.add(code)

        return [c for c in codes if c in soe_codes]

    def _filter_dividends(self, codes: list) -> list:
        """过滤：连续5年现金分红"""
        current_year = datetime.now().year
        required_years = set(range(current_year - 5, current_year))  # 5年

        passed = []
        for i, code in enumerate(codes):
            if i % 20 == 0:
                self._report(f"检查分红: {i}/{len(codes)}", "分红", len(codes) - i)
            div_df = df.get_dividend_history(code)
            if div_df.empty:
                continue
            # 检查连续5年有现金分红
            cash_years = set()
            for _, row in div_df.iterrows():
                report_date = str(row.get('报告期', ''))
                if len(report_date) < 4:
                    continue
                year = int(report_date[:4])
                # 检查是否有现金分红（每股派息 > 0）
                cash_div = _safe_float(row.get('现金分红-每股派息', 0))
                if cash_div <= 0:
                    cash_div = _safe_float(row.get('现金分红-现金分红比例', 0))
                if cash_div > 0:
                    cash_years.add(year)
            if required_years.issubset(cash_years):
                passed.append(code)

        return passed

    def _filter_no_issuance(self, codes: list) -> list:
        """过滤：近5年无增发或发债"""
        five_years_ago = (pd.Timestamp.now() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
        code_set = set(codes)

        # 检查增发
        try:
            issuance_df = df.get_additional_issuance()
            if not issuance_df.empty:
                # 寻找日期列
                date_col = None
                for col in issuance_df.columns:
                    if '日期' in col or '时间' in col:
                        date_col = col
                        break
                code_col = None
                for col in issuance_df.columns:
                    if '代码' in col:
                        code_col = col
                        break
                if date_col and code_col:
                    recent = issuance_df[issuance_df[date_col].astype(str) >= five_years_ago]
                    issuers = set(str(c).zfill(6) for c in recent[code_col])
                    code_set -= issuers
        except Exception as e:
            logger.warning(f"增发数据检查失败: {e}")

        # 检查可转债
        try:
            bond_df = df.get_convertible_bonds()
            if not bond_df.empty:
                date_col = None
                for col in bond_df.columns:
                    if '日期' in col or '时间' in col:
                        date_col = col
                        break
                code_col = None
                for col in bond_df.columns:
                    if '代码' in col:
                        code_col = col
                        break
                if date_col and code_col:
                    recent_bonds = bond_df[bond_df[date_col].astype(str) >= five_years_ago]
                    bond_codes = set(str(c).zfill(6)[:6] for c in recent_bonds[code_col])
                    code_set -= bond_codes
        except Exception as e:
            logger.warning(f"可转债数据检查失败: {e}")

        return [c for c in codes if c in code_set]

    def _filter_buyback(self, codes: list) -> list:
        """过滤：大股东主动回购"""
        buyback_df = df.get_buyback_data()
        if buyback_df.empty:
            logger.warning("无法获取回购数据")
            return codes

        buyback_codes = set()
        for _, row in buyback_df.iterrows():
            raw_code = str(row.get('股票代码', '')).strip()
            code = raw_code.replace('sh', '').replace('sz', '').zfill(6)
            progress = str(row.get('实施进度', ''))
            # 已完成或正在实施的回购
            if any(kw in progress for kw in ['完成', '实施中', '实施', '董事会预案', '股东大会通过']):
                buyback_codes.add(code)

        return [c for c in codes if c in buyback_codes]

    # ============================
    #  Phase 2: 逐只检查函数
    # ============================

    def _check_dividend_yield(self, code: str) -> bool:
        """股息率 > 4%

        Returns:
            True: 符合条件（股息率>=4%）
            False: 不符合条件（股息率<4% 或 无分红）
            None: 无法判断（API失败）
        """
        try:
            info = df.get_stock_info(code)
            if info.empty:
                logger.debug(f"{code}: 无法获取股票信息，跳过")
                return None  # API失败，跳过

            # 获取最新价
            price = None
            for _, row in info.iterrows():
                if str(row.get('item', '')) in ['最新价', '最新', '股价']:
                    price = _safe_float(row.get('value'))
                    break
            if not price or price <= 0:
                logger.debug(f"{code}: 无法获取有效价格，跳过")
                return None  # 数据异常，跳过

            # 获取最近一年分红
            div_df = df.get_dividend_history(code)
            if div_df.empty:
                return False  # 无分红记录，不符合条件

            current_year = datetime.now().year
            total_div = 0.0
            for _, row in div_df.iterrows():
                year_str = str(row.get('报告期', ''))[:4]
                if not year_str.isdigit():
                    continue
                year = int(year_str)
                if year >= current_year - 1:
                    per_share = _safe_float(row.get('现金分红-每股派息', 0))
                    if per_share > 0:
                        total_div += per_share

            if total_div <= 0:
                return False  # 无现金分红，不符合条件

            yield_pct = (total_div / price) * 100
            return yield_pct >= 4.0
        except Exception as e:
            logger.debug(f"dividend_yield check {code}: {e}")
            return None  # 异常时跳过，避免误杀

    def _check_3year_growth(self, code: str) -> bool:
        """连续3年年报营收净利增长

        Returns:
            True: 符合条件
            False: 不符合条件
            None: 无法判断（API失败）
        """
        try:
            profit_df = df.get_profit_statement(code)
            if profit_df.empty:
                logger.debug(f"{code}: 无法获取利润表，跳过")
                return None  # API失败，跳过

            # 取年报数据
            annual = profit_df[profit_df['报告日'].astype(str).str.endswith('1231')].head(4)
            if len(annual) < 4:
                return False  # 数据不足，不符合条件

            for i in range(3):
                curr_rev = _safe_float(annual.iloc[i].get('营业总收入', 0))
                prev_rev = _safe_float(annual.iloc[i + 1].get('营业总收入', 0))
                curr_profit = _safe_float(annual.iloc[i].get('净利润', 0))
                prev_profit = _safe_float(annual.iloc[i + 1].get('净利润', 0))

                if prev_rev <= 0 or prev_profit <= 0:
                    return False
                if curr_rev <= prev_rev or curr_profit <= prev_profit:
                    return False

            return True
        except Exception as e:
            logger.debug(f"3year_growth check {code}: {e}")
            return None  # 异常时跳过

    def _check_quarterly_growth(self, code: str) -> bool:
        """季报营收净利同比、环比增长

        Returns:
            True: 符合条件
            False: 不符合条件
            None: 无法判断（API失败）
        """
        try:
            profit_df = df.get_profit_statement(code)
            if profit_df.empty or len(profit_df) < 6:
                logger.debug(f"{code}: 季报数据不足，跳过")
                return None  # 数据不足，跳过

            # 最新一季
            latest = profit_df.iloc[0]
            prev_q = profit_df.iloc[1]  # 上一季（环比）

            # 去年同期（同比）—— 找相同季度
            latest_date = str(latest.get('报告日', ''))
            quarter_suffix = latest_date[4:]  # 如 "0331", "0630" 等
            yoy_row = None
            for i in range(1, len(profit_df)):
                d = str(profit_df.iloc[i].get('报告日', ''))
                if d.endswith(quarter_suffix) and d[:4] != latest_date[:4]:
                    yoy_row = profit_df.iloc[i]
                    break

            if yoy_row is None:
                return False  # 找不到同比数据，不符合条件

            curr_rev = _safe_float(latest.get('营业总收入'))
            prev_q_rev = _safe_float(prev_q.get('营业总收入'))
            yoy_rev = _safe_float(yoy_row.get('营业总收入'))

            curr_profit = _safe_float(latest.get('净利润'))
            prev_q_profit = _safe_float(prev_q.get('净利润'))
            yoy_profit = _safe_float(yoy_row.get('净利润'))

            # 同比增长
            if yoy_rev <= 0 or yoy_profit <= 0:
                return False
            if curr_rev <= yoy_rev or curr_profit <= yoy_profit:
                return False
            # 环比增长
            if prev_q_rev <= 0 or prev_q_profit <= 0:
                return False
            if curr_rev <= prev_q_rev or curr_profit <= prev_q_profit:
                return False

            return True
        except Exception as e:
            logger.debug(f"quarterly_growth check {code}: {e}")
            return None  # 异常时跳过

    def _check_controller_stable(self, code: str) -> bool:
        """实控人未变更"""
        try:
            ctrl_df = df.get_controller_info()
            if ctrl_df.empty:
                return True

            stock_rows = ctrl_df[
                ctrl_df['证券代码'].astype(str).str.replace('sh', '').str.replace('sz', '').str.zfill(6) == code
            ]
            if stock_rows.empty:
                return True  # 无记录视为稳定

            # 如果实控人只有一个唯一值则稳定
            controllers = stock_rows['实际控制人名称'].dropna().unique()
            return len(controllers) <= 1
        except Exception as e:
            logger.debug(f"controller_stable check {code}: {e}")
            return True

    def _check_cash_gt_debt(self, code: str) -> bool:
        """货币资金 > (短期借款+长期借款+应付债券+担保+质押)

        Returns:
            True: 符合条件
            False: 不符合条件
            None: 无法判断（API失败）
        """
        try:
            bs = df.get_balance_sheet(code)
            if bs.empty:
                logger.debug(f"{code}: 无法获取资产负债表，跳过")
                return None  # API失败，跳过

            latest = bs.iloc[0]
            cash = _safe_float(latest.get('货币资金', 0))
            short_debt = _safe_float(latest.get('短期借款', 0))
            long_debt = _safe_float(latest.get('长期借款', 0))
            bonds = _safe_float(latest.get('应付债券', 0))

            # 质押数据
            pledge_amount = 0
            try:
                pledge_df = df.get_pledge_data()
                if not pledge_df.empty:
                    code_col = None
                    for col in pledge_df.columns:
                        if '代码' in col:
                            code_col = col
                            break
                    if code_col:
                        stock_pledge = pledge_df[
                            pledge_df[code_col].astype(str).str.zfill(6) == code
                        ]
                        if not stock_pledge.empty:
                            # 质押比例
                            ratio_col = None
                            for col in stock_pledge.columns:
                                if '比例' in col or '比率' in col:
                                    ratio_col = col
                                    break
                            if ratio_col:
                                ratio = _safe_float(stock_pledge.iloc[0].get(ratio_col, 0))
                                # 质押比例超过30%认为风险较大，粗略估算
                                if ratio > 30:
                                    return False
            except Exception:
                pass

            total_debt = short_debt + long_debt + bonds + pledge_amount
            return cash > total_debt
        except Exception as e:
            logger.debug(f"cash_gt_debt check {code}: {e}")
            return None  # 异常时跳过
