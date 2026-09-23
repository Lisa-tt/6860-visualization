"""Write the end-to-end analysis workflow document from current outputs."""

import hashlib
from pathlib import Path


def _audit_value(cleaning, check, column="records_flagged"):
    row = cleaning.loc[cleaning["check"].eq(check)]
    return row.iloc[0][column] if not row.empty else float("nan")


def _sha256(path):
    """Return a stable identifier for the local source snapshot."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_analysis_workflow(data, tables, audit_tables, project_root):
    """Create a Chinese Markdown record of the reproducible analysis process."""
    project_root = Path(project_root)
    source_path = project_root / "data" / "Education-services-with-station-access_loc.csv"
    population_path = (
        project_root
        / "data"
        / "external"
        / "abs"
        / "2021_GCP_RA_for_AUS_short-header.zip"
    )
    source_size_mb = source_path.stat().st_size / (1024 * 1024)
    source_sha256 = _sha256(source_path)
    population_size_mb = population_path.stat().st_size / (1024 * 1024)
    population_sha256 = _sha256(population_path)
    cleaning = audit_tables["cleaning_summary"]
    missing = audit_tables["missingness_summary"].set_index("variable")
    samples = audit_tables["analysis_sample_flow"].set_index("analysis_sample")

    state = tables["state_summary"]
    remote = tables["remoteness_summary"]
    qa = tables["quality_area_summary"]
    transport = tables["transport_by_remoteness"]
    operations = tables["operations_by_remoteness"]
    quality_models = tables["quality_model_coefficients"]
    operational_models = tables["operational_model_effects"]
    compound = tables["compound_disadvantage"]
    coverage = tables["population_coverage_by_remoteness"]
    national_coverage = tables["population_coverage_national"].iloc[0]
    descriptive = tables["eda_descriptive_statistics"]

    highest_state = state.loc[state["below_nqs_pct_of_rated"].idxmax()]
    weakest_qa = qa.loc[qa["below_nqs_pct_of_rated"].idxmax()]
    remote_low = remote.loc[remote["below_nqs_pct_of_rated"].idxmin()]
    remote_high = remote.loc[remote["below_nqs_pct_of_rated"].idxmax()]
    city_transport = transport[
        transport["remoteness_area"].eq("Major Cities of Australia")
        & transport["transport_mode"].eq("Nearest listed mode")
    ].iloc[0]
    very_remote_transport = transport[
        transport["remoteness_area"].eq("Very Remote Australia")
        & transport["transport_mode"].eq("Nearest listed mode")
    ].iloc[0]
    centre_operations = operations[
        operations["service_type"].eq("Centre-Based Care")
    ]
    capacity_high = centre_operations.loc[
        centre_operations["median_approved_places"].idxmax()
    ]
    capacity_low = centre_operations.loc[
        centre_operations["median_approved_places"].idxmin()
    ]
    bus_quality = quality_models[
        quality_models["transport_mode"].eq("Bus")
    ].iloc[0]
    train_quality = quality_models[
        quality_models["transport_mode"].eq("Train")
    ].iloc[0]
    capacity_bus = operational_models[
        operational_models["outcome"].eq("Centre-Based approved capacity")
        & operational_models["transport_mode"].eq("Bus")
    ].iloc[0]
    flagged_groups = compound[compound["enhanced_screening_flag"]]
    city_coverage = coverage[
        coverage["remoteness_area"].eq("Major Cities of Australia")
    ].iloc[0]
    very_remote_coverage = coverage[
        coverage["remoteness_area"].eq("Very Remote Australia")
    ].iloc[0]

    service_type_counts = data["ServiceType"].value_counts()
    matched = int(data["remoteness_area"].notna().sum())
    rated = int(data["is_rated"].sum())
    annual_hours = int(data["annual_weekly_operating_hours"].notna().sum())
    duplicate_rows = int(_audit_value(cleaning, "Exact duplicate rows"))
    duplicate_ids = int(_audit_value(cleaning, "Duplicate ServiceApprovalNumber"))
    transport_review = int(
        _audit_value(cleaning, "Bus or train distance over 500 km")
    )
    overlap_adjusted = int(
        _audit_value(cleaning, "Term calendars with overlapping sessions")
    )
    descriptive_markdown = [
        "| Variable | Scope | Valid n | Missing % | Mean | SD | Q1 | Median | Q3 | P90 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in descriptive.itertuples(index=False):
        descriptive_markdown.append(
            f"| {row.measure} | {row.scope} | {int(row.valid_n):,} | "
            f"{row.missing_pct:.1f} | {row.mean:.1f} | {row.std_dev:.1f} | "
            f"{row.q1:.1f} | {row.median:.1f} | {row.q3:.1f} | {row.p90:.1f} |"
        )
    descriptive_markdown = "\n".join(descriptive_markdown)

    lines = [
        "# ACECQA 教育与保育服务数据分析工作流",
        "",
        "> 本文档记录本项目从数据获取、数据审计、数据清洗、EDA、统计分析、可视化到结论与决策建议的完整流程。文中的数字来自当前项目内固定的数据快照；重新运行 `python run_analysis.py` 会同步更新分析表、图表和本文档。",
        "",
        "## 1. 项目目的与核心问题",
        "",
        "本项目面向 ACECQA 及澳大利亚各级政府的政策和质量管理场景。核心业务问题为：",
        "",
        "> 澳大利亚教育和保育服务中，质量、可达性与运营能力的差距在哪里重叠？哪些地理群体值得 ACECQA 或政府进一步调查？",
        "",
        "分析单位是一个获批教育与保育服务机构，而不是 provider。分析采用以下递进逻辑：",
        "",
        "```text",
        "National landscape -> Quality -> Accessibility & Coverage",
        "-> Quality x Accessibility -> Operations -> Synthesis",
        "```",
        "",
        "本项目的目标是形成可验证的筛查证据，而不是直接证明因果关系或据此分配资源。",
        "",
        "## 2. 数据源与数据获取",
        "",
        "### 2.1 ACECQA 教育服务数据",
        "",
        "- 本地文件：`data/Education-services-with-station-access_loc.csv`。",
        "- 官方来源：[ACECQA National Registers](https://www.acecqa.gov.au/resources/national-registers)。该页面说明 approved services 数据可导出为 CSV，并由 NQA ITS 每日更新。",
        "- 本项目不在运行时重新下载每日更新的数据，而是使用提交目录中的固定快照保证复现。",
        f"- 当前源文件大小：**{source_size_mb:.2f} MB**；SHA-256：`{source_sha256}`。",
        "- 教师增加的空间与交通字段包括 Geometry、最近 train/bus station 及相应距离，因此这些字段需要额外的数据质量审查。",
        f"- 当前快照包含 **{len(data):,}** 个服务，Centre-Based Care 为 **{int(service_type_counts.get('Centre-Based Care', 0)):,}** 个，Family Day Care 为 **{int(service_type_counts.get('Family Day Care', 0)):,}** 个。",
        "",
        "### 2.2 ABS 外部地理数据",
        "",
        "- 本地文件：`data/external/abs/RA_2021_AUST_GDA2020/RA_2021_AUST_GDA2020.shp`。",
        "- 来源：[Australian Bureau of Statistics, ASGS Edition 3 Digital Boundary Files](https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs/edition-3-july-2021-june-2026/access-and-downloads/digital-boundary-files)。",
        "- 用途：将服务坐标分类为 Major Cities、Inner Regional、Outer Regional、Remote 和 Very Remote。",
        "- 方法：使用 GeoPandas 点在面空间连接，而不是用 postcode 近似替代坐标。",
        "",
        "### 2.3 ABS 2021 儿童人口 denominator",
        "",
        "- 本地文件：`data/external/abs/2021_GCP_RA_for_AUS_short-header.zip`。",
        "- 官方来源：[ABS 2021 Census DataPacks](https://www.abs.gov.au/census/find-census-data/datapacks)，选择 General Community Profile、Remoteness Area、Australia。",
        "- 使用表 G04A 的逐岁 Persons 字段 `Age_yr_0_P` 至 `Age_yr_13_P`，精确加总 0-13 岁 usual residents，不对 10-14 岁进行比例拆分。",
        "- 使用 State-specific RA code 与服务点所在 boundary RA 一对一连接；人口分析使用 boundary state，而不是可能与边界不一致的 source State。",
        f"- DataPack 大小：**{population_size_mb:.2f} MB**；SHA-256：`{population_sha256}`。",
        f"- 纳入常规八州/领地 RA 的儿童人口为 **{int(national_coverage['child_population_0_13']):,}**。",
        "",
        "### 2.4 未纳入的数据",
        "",
        "当前仍没有入托需求、实际 enrolment、空位、等候名单、staffing、公交频次或出行时间数据。因此即使加入人口 denominator，也不能把较低 places-per-child 直接解释为 unmet demand，不能使用 underserved 作为确定性结论。",
        "",
        "## 3. 数据理解与变量角色",
        "",
        "| 分析维度 | 主要变量 | 分析角色 |",
        "|---|---|---|",
        "| 地理 | State、坐标、ABS Remoteness Area | 主分析层级 |",
        "| 质量 | OverallRating、QA1-QA7、Meeting+、Below NQS | 结果变量 |",
        "| 可达性 | Bus distance、Train distance、Nearest transport distance | 代理指标 |",
        "| 覆盖率 | 0-13 岁人口、services/1,000、approved places/1,000 | denominator 与标准化指标 |",
        "| 运营 | Approved places、annual weekly hours、approval cohort | 结果变量 |",
        "| 服务结构 | broad ServiceType、detailed offerings | 混杂解释或分层变量 |",
        "",
        "质量评级是有序类别，不假设类别间距相等，因此不计算平均 rating score。详细服务 offering 是多标签变量，一个机构可同时提供多类服务，不能把 offering counts 相加后解释为市场份额。",
        "",
        "## 4. 数据审计",
        "",
        f"- 数据行数：**{len(data):,}**。",
        f"- 完全重复行：**{duplicate_rows:,}**；重复 Service Approval Number：**{duplicate_ids:,}**。",
        f"- 有 recognised overall rating：**{rated:,}**，占 **{rated / len(data) * 100:.1f}%**。",
        f"- 有效坐标并成功匹配 ABS RA：**{matched:,}**，占 **{matched / len(data) * 100:.1f}%**。",
        f"- Overall rating 缺失率：**{missing.loc['OverallRating', 'missing_pct']:.1f}%**。",
        f"- Annual weekly hours 缺失率：**{missing.loc['annual_weekly_operating_hours', 'missing_pct']:.1f}%**。",
        f"- 超过 500 km、需要检查的 bus/train 距离记录：**{transport_review:,}**。这些观测经偏远地区核查后保留，不使用 IQR 规则机械删除。",
        f"- 存在重叠 school-term sessions 并完成区间合并的记录：**{overlap_adjusted:,}**。",
        "",
        "完整审计输出位于 `outputs/audit/`，包括 missingness、category audit、cleaning decisions、transport anomaly review 和 analysis sample flow。",
        "",
        "## 5. 数据清洗与准备",
        "",
        "### 5.1 文本、类别与标识符",
        "",
        "1. 去除字段名和文本值两侧空格，空字符串转为缺失。",
        "2. State 统一为大写；postcode 保留字符串并补足四位。",
        "3. 检查 ServiceApprovalNumber 唯一性，不在没有证据时自动去重。",
        "4. Overall 与 QA1-QA7 仅接受规定的 NQS rating categories；无法识别的值只退出对应质量分母。",
        "",
        "### 5.2 数值与日期",
        "",
        "1. Approved places、bus distance、train distance 使用显式 numeric coercion，并记录转换失败。",
        "2. Approval date 和 rating date 按 day-first 规则解析，派生 approval year、rating year 和 approval cohort。",
        "3. Approval year 仅描述当前注册服务的 approval cohort，不解释为行业历史增长。",
        "",
        "### 5.3 质量派生变量",
        "",
        "- `is_rated`：Overall rating 为 recognised category。",
        "- `is_below_nqs`：Significant Improvement Required 或 Working Towards NQS。",
        "- `is_meeting_or_above`：Meeting、Exceeding 或 Excellent。",
        "- 缺失 rating 的服务保留在空间、可达性和运营分析中，仅从相应质量分母排除。",
        "",
        "### 5.4 交通距离",
        "",
        "1. 负数或缺失距离只从受影响的交通分析中排除。",
        "2. `nearest_transport_distance_km` 取有效 bus/train 距离中的较小值。",
        "3. 超过 500 km 的观测标记为 review，不自动删除，因为偏远地区可能存在真实极端距离。",
        "4. Bus 与 Train 在主要分析和模型中分别保留，避免把没有铁路等同于没有任何公共交通。",
        "",
        "### 5.5 营业时间",
        "",
        "1. 将每日 start/end time 转为分钟，允许跨午夜。",
        "2. Annual、school-term、holiday calendars 分别计算，不相加。",
        "3. School-term 同一天多 session 使用时间区间并集，避免重叠重复计算。",
        f"4. 可用于主要 annual-hours 分析的记录为 **{annual_hours:,}**；完整模型样本为 **{int(samples.loc['Annual-hours analysis', 'services']):,}**。",
        "",
        "### 5.6 地理准备",
        "",
        "1. 从 `c(longitude, latitude)` 解析坐标，并用澳大利亚及外部领地的宽边界检查明显错误。",
        "2. 服务点以 EPSG:4326 建立，转换到 ABS boundary CRS 后执行 `geopandas.sjoin(..., predicate='within')`。",
        "3. 保留 source State，不用 boundary State 静默覆盖；不一致记录单独标记。",
        "4. 全国网格汇总使用 EPSG:3577 Australian Albers，避免直接在经纬度上创建等距离网格。",
        "",
        "### 5.7 人口 denominator 与覆盖率",
        "",
        "1. 校验 G04A 中 0-13 岁逐岁人口字段完整、非负，并仅保留八州/领地的常规 Remoteness Areas。",
        "2. 以 `RA_CODE21` 连接服务 numerator 与人口 denominator，避免用同名 remoteness category 跨州误合并。",
        "3. 主要覆盖率指标为 Centre-Based approved places / 0-13 population × 1,000；同时输出全部 services 和 Centre-Based services per 1,000。",
        "4. Family Day Care 没有可比 approved-place 字段，因此只进入 service-count ratio，不进入 approved-place ratio。",
        "5. Census 2021 与当前 register 存在时间错配；比率用于 broad planning screen，不是当前空位率。",
        "6. 0-13 岁人口少于 5,000 的 State x RA 单元标记为 small denominator；保留观察值，但不据此单独作资源决策。",
        "",
        "## 6. 缺失值与分析样本策略",
        "",
        "本项目采用 analysis-specific deletion，不执行全表 complete-case deletion，也不对 rating、capacity 或 hours 进行无依据插补。",
        "",
        "| 分析样本 | 服务数 | 占源数据 |",
        "|---|---:|---:|",
    ]
    for sample_name, row in audit_tables["analysis_sample_flow"].set_index(
        "analysis_sample"
    ).iterrows():
        lines.append(
            f"| {sample_name} | {int(row['services']):,} | {row['share_of_source_pct']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "关键口径：质量百分比以 rated services 为分母；capacity 只比较 Centre-Based Care；annual hours 只使用 annual calendar；筛查图的质量、交通与容量全部使用兼容的 Centre-Based denominator。",
            "",
            "## 7. 探索性数据分析（EDA）",
            "",
            "EDA 的作用是检查数据结构、分布、异常值、样本不平衡和原始关系，从而决定后续图表与模型。EDA 图不直接替代最终政策展示图。",
            "",
            "### EDA 1：数据完整性与样本形成",
            "",
            "![EDA 1](outputs/figures/eda/E01_data_readiness.png)",
            "",
            "主要洞察：空间和交通变量覆盖率高，但 annual、school-term 和 holiday hours 缺失明显；因此运营分析不能与全国服务总量使用同一个分母。质量模型、capacity 和 hours 模型也必须分别报告 n。",
            "",
            "### EDA 2：单变量分布",
            "",
            "![EDA 2](outputs/figures/eda/E02_univariate_profile.png)",
            "",
            "主要洞察：交通距离高度右偏，适合使用 log display、median 和 robust quantiles；capacity 存在合理的大型中心尾部；营业时间的分布随 broad service type 改变，后续模型需要控制 service type。图中 P99 仅用于显示范围，记录没有从分析数据中删除。",
            "",
            "### EDA 3：地理样本结构",
            "",
            "![EDA 3](outputs/figures/eda/E03_geographic_sample_structure.png)",
            "",
            "主要洞察：不同州的 remoteness composition 差异很大，Remote 和 Very Remote 的部分 State x RA cells 样本很小。州级或偏远程度的 raw quality comparisons 可能受到结构混杂，因此最终模型同时控制 State、Remoteness 和 broad Service Type。",
            "",
            "### EDA 4：原始运营关系",
            "",
            "![EDA 4](outputs/figures/eda/E04_raw_relationship_diagnostics.png)",
            "",
            "主要洞察：服务层面的 transport distance 与 capacity/hours 呈现高密度重叠和明显异方差，单纯相关系数或普通散点图都可能误导。因此使用 hexbin 减少 overplotting，并在正式分析中使用 log-distance、geographic controls 和 robust covariance。",
            "",
            "### EDA 5：关键描述性统计表",
            "",
            "![EDA 5](outputs/figures/eda/E05_descriptive_statistics_table.png)",
            "",
            descriptive_markdown,
            "",
            "主要洞察：距离与 capacity 的 mean 明显受右尾影响，因此正文优先报告 median、quartiles 和 P90；annual hours 的有效样本较少，所有运营结论必须附带有效 n。完整精度表位于 `outputs/tables/eda_descriptive_statistics.csv`。",
            "",
            "## 8. 正式分析方法",
            "",
            "### 8.1 National landscape",
            "",
            "使用等面积网格和官方 ABS outline 描述 observed provision；该图仍然只展示服务位置。人口标准化另以 State-specific RA 汇总，避免把网格 service count 误当作需求覆盖率。",
            "",
            "### 8.2 Quality",
            "",
            "使用 100% rating composition 和 Meeting+ benchmark gaps，分别比较 State、Remoteness、Overall 及 QA1-QA7。避免把 ordinal ratings 转成任意平均分。",
            "",
            "### 8.3 Accessibility & coverage",
            "",
            "Bus、Train 和 nearest listed mode 使用 median、P90 和 log scale 描述。Distance 是 proximity proxy，不代表班次、票价、换乘或家庭真实出行。",
            f"Coverage 使用 Centre-Based approved places per 1,000 children aged 0-13；全国 benchmark 为 **{national_coverage['approved_places_per_1000_children']:.1f}**。同时报告 services per 1,000，以区分机构数量和机构规模。",
            "",
            "### 8.4 Quality x accessibility",
            "",
            "使用 Binomial GLM 和 HC1 robust covariance：",
            "",
            "```text",
            "Meeting+ ~ log2(distance + 0.1 km) + remoteness + broad service type + state",
            "```",
            "",
            "Bus 与 Train 分开估计；输出 odds ratio、95% CI、样本量和 adjusted marginal predictions。模型用于评估 adjusted association，不用于因果推断。",
            "",
            "### 8.5 Operations",
            "",
            "Capacity 限制为 Centre-Based Care，并对 `log(approved places)` 使用 HC3 robust OLS；annual weekly hours 使用 HC3 robust OLS，并额外控制 broad service type。",
            "",
            "```text",
            "log(capacity) ~ log2(distance + 0.1 km) + remoteness + state",
            "annual hours ~ log2(distance + 0.1 km) + remoteness + broad service type + state",
            "```",
            "",
            "### 8.6 Multidimensional synthesis",
            "",
            "筛查单位为 Centre-Based State x Remoteness group，要求至少 20 个 rated services。保留原三项规则用于审计，并新增第四项人口规则：Below-NQS rate 高于全国 benchmark、transport distance 高于全国 median、median approved capacity 低于全国 median、approved places per 1,000 children 低于全国 benchmark。增强筛查不产生加权分数，也不证明 disadvantage。",
            "",
            "## 9. 最终可视化设计",
            "",
            "- `outputs/figures/presentation/`：8 张用于 5-10 分钟叙事的核心静态图，其中 `04b` 为人口标准化覆盖率。",
            "- `outputs/figures/appendix/`：offering、未调整 quintiles 和 approval cohorts 等支持图。",
            "- `outputs/figures/eda/`：5 张数据质量、分布、原始关系和描述性统计图。",
            "- `outputs/interactive/`：空间、质量、人口覆盖率和多维筛查的 4 个独立 HTML explorer。",
            "",
            "图表选择遵循：构成比较使用 100% stacked bars；benchmark 使用以 0 为中心的 diverging scale；右偏距离使用 log scale 和 robust quantiles；大量点使用 grid/hexbin；小样本质量率使用 n 或 Wilson interval；普通 bar chart 坚持零基线。",
            "",
            "## 10. 主要结论",
            "",
            f"1. **地理结构明显。** Nearest listed transport 的中位数从 Major Cities 的 **{city_transport['median_distance_km']:.2f} km** 上升到 Very Remote 的 **{very_remote_transport['median_distance_km']:.2f} km**。",
            f"2. **质量差异存在，但属于未调整比较。** {highest_state['state']} 的 Below-NQS share 最高，为 **{highest_state['below_nqs_pct_of_rated']:.1f}%**；Remoteness groups 从 {remote_low['below_nqs_pct_of_rated']:.1f}% 到 {remote_high['below_nqs_pct_of_rated']:.1f}% 不等。",
            f"3. **全国最弱质量领域为 {weakest_qa['quality_area']}。** 其 Below-NQS rate 为 **{weakest_qa['below_nqs_pct_of_rated']:.1f}%**。",
            f"4. **Raw quality-accessibility gradient 经调整后明显减弱。** Bus distance 每翻倍 OR 为 **{bus_quality['odds_ratio_per_distance_doubling']:.3f}**（95% CI {bus_quality['or_ci95_lower']:.3f}-{bus_quality['or_ci95_upper']:.3f}）；Train OR 为 **{train_quality['odds_ratio_per_distance_doubling']:.3f}**（95% CI {train_quality['or_ci95_lower']:.3f}-{train_quality['or_ci95_upper']:.3f}）。",
            f"5. **Capacity 的地理差异比 opening-hours association 更清楚。** Centre-Based median capacity 从 {capacity_high['median_approved_places']:.0f} places 降至 {capacity_low['median_approved_places']:.0f} places；adjusted bus-distance effect 为 **{capacity_bus['adjusted_effect_per_distance_doubling']:.2f}%** per doubling。",
            f"6. **人口标准化后 capacity intensity 仍呈梯度。** Approved places per 1,000 children 从 Major Cities 的 **{city_coverage['approved_places_per_1000_children']:.1f}** 降至 Very Remote 的 **{very_remote_coverage['approved_places_per_1000_children']:.1f}**。",
            f"7. **共有 {len(flagged_groups)} 个 State x Remoteness groups 同时触发四项增强筛查规则。** 这些是进一步调查对象，不是确定性的资源优先级。",
            "",
            "## 11. 结论",
            "",
            "现有证据支持一个以 geography 为主线的判断：服务位置、交通距离、质量构成和 capacity 在地理上并不均匀；加入儿童人口 denominator 后，Very Remote 的 approved-place intensity 仍明显低于 Major Cities。但质量与交通距离的 raw relationship 很大程度上与 remoteness、state 和 service composition 同时变化。ACECQA 不应把 proximity、coverage 或 quality 的简单相关解释为因果关系。更稳妥的做法是把四维筛查结果作为定向数据核查和地方调查的入口。",
            "",
            "## 12. 决策建议",
            "",
            f"1. **对 {len(flagged_groups)} 个增强筛查群体开展第二阶段验证。** 核查 rating denominator、具体 QA 弱项、当地 service mix、capacity 使用率和真实出行条件，再决定是否需要政策干预。",
            "2. **把 Census denominator 升级为同年人口与年龄适配需求。** 后续应接入与 register 同期的小区域 ERP，并按 Long Day Care、preschool、OSHC 的目标年龄分别构造 denominator。",
            "3. **提升可达性数据质量。** 加入 travel time、public transport frequency、service span、affordability、道路网络和家庭出行方式，而不是只使用直线或最近站点距离。",
            f"4. **针对 {weakest_qa['quality_area']} 和地理差异开展质量支持。** 先分析该 QA 在高风险州和偏远地区的具体标准项，再设计 guidance、training 或 regulatory follow-up。",
            "5. **改进运营数据完整性。** Annual hours 当前并非全覆盖；应统一 calendar reporting，并避免把 annual、school-term 和 holiday hours 相加。",
            "6. **维持 service-type-aware 比较。** Capacity 继续限制在 Centre-Based Care；质量和 hours 模型保留 broad Service Type，防止把服务结构差异误判为地理效应。",
            "",
            "## 13. 局限性与解释边界",
            "",
            "- 当前 register 是 cross-sectional snapshot，不能识别趋势或因果机制。",
            "- 2021 Census denominator 与当前 register 存在时间错配；Census 小区域计数还经过 confidentiality perturbation。",
            "- 0-13 岁总人口并不等同于每种 service offering 的 eligible population 或实际需求。",
            "- Approved places 只覆盖 Centre-Based Care，且不是 enrolment、vacancies、staffing 或 waitlist。",
            "- RA 聚合会隐藏区域内部差异，结果可能受可变空间单元问题（MAUP）影响。",
            "- 儿童人口少于 5,000 的 State x RA 比率对少量 numerator 变化很敏感，必须同时查看原始人口与 approved places。",
            "- Remoteness 是服务所在地，不是家庭所在地或 catchment。",
            "- 小型 Remote/Very Remote groups 的百分比不稳定，必须结合 n 和 CI。",
            "- Detailed offerings 可重叠，不能作为互斥服务市场份额。",
            "- 筛查规则用于 prioritisation for investigation，不是对 disadvantage 的最终认定。",
            "",
            "## 14. 复现方式",
            "",
            "```powershell",
            "cd E:\\USYD\\6860\\project1",
            "python -m pip install -r requirements.txt",
            "python scripts\\download_abs_boundaries.py",
            "python scripts\\download_abs_child_population.py",
            "python run_analysis.py",
            "python -m unittest discover -s tests -v",
            "```",
            "",
            "主要可审计输出：",
            "",
            "- `outputs/audit/`：清洗、missingness、category 和样本流程。",
            "- `outputs/tables/`：描述性结果、模型系数、预测和筛查表。",
            "- `outputs/report/key_findings.md`：按分析逻辑整理的结果。",
            "- `outputs/report/code_audit.md`：保留、修改和替换决策。",
            "- `outputs/data/service_remoteness_classification.csv`：服务级地理分类结果。",
        ]
    )

    output_path = project_root / "ANALYSIS_WORKFLOW.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
