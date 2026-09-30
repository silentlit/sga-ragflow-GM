#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import json
import logging
from abc import ABC
from copy import deepcopy
from agent.tools.base import ToolBase, ToolParamBase, ToolMeta


def calculate_formulation(
    target_mass_kg: float = 50.0,
    ni_ratio: float = 0.92,
    co_ratio: float = 0.04,
    mn_ratio: float = 0.04,
    zr_mol: float = 0.002,
    li_tm_ratio: float = 1.03
) -> dict:
    """
    计算超高镍三元正极固相烧结所需原料的投料重量（前驱体、单水氢氧化锂、氧化锆）

    :param target_mass_kg: 目标合成正极材料成品重量 (kg)
    :param ni_ratio: 镍摩尔比例 (Ni mol ratio, 如 0.92)
    :param co_ratio: 钴摩尔比例 (Co mol ratio, 如 0.04)
    :param mn_ratio: 锰摩尔比例 (Mn mol ratio, 如 0.04)
    :param zr_mol: 二氧化锆掺杂摩尔比 (如 0.002 代表 0.2 mol%)
    :param li_tm_ratio: 锂与过渡金属投料摩尔比 (Li/TM ratio, 如 1.03)
    :return: 包含投料千克数、摩尔量及推荐工艺制度的字典
    """
    # 归一化过渡金属摩尔比
    tm_sum = ni_ratio + co_ratio + mn_ratio
    if tm_sum <= 0:
        raise ValueError("过渡金属比例总和必须大于 0")
    ni = ni_ratio / tm_sum
    co = co_ratio / tm_sum
    mn = mn_ratio / tm_sum

    # 各元素原子摩尔质量 (g/mol)
    mw_ni = 58.6934
    mw_co = 58.9332
    mw_mn = 54.9380
    mw_o = 15.9994
    mw_h = 1.0080
    mw_li = 6.941
    mw_zr = 91.224

    # 化合物分子量
    # 前驱体 Ni_xCo_yMn_z(OH)2
    mw_precursor = ni * mw_ni + co * mw_co + mn * mw_mn + 2 * (mw_o + mw_h)
    # 单水氢氧化锂 LiOH·H2O
    mw_lioh = mw_li + mw_o + mw_h + (2 * mw_h + mw_o)
    # 二氧化锆 ZrO2
    mw_zro2 = mw_zr + 2 * mw_o
    # 成品正极估算分子量 Li_1.0(NiCoMn)O2
    mw_product = mw_li + (ni * mw_ni + co * mw_co + mn * mw_mn) + 2 * mw_o

    # 目标合成摩尔数 (mol)
    target_grams = target_mass_kg * 1000.0
    product_moles = target_grams / mw_product

    # 理论投料计算
    precursor_kg = (product_moles * mw_precursor) / 1000.0
    lioh_kg = (product_moles * li_tm_ratio * mw_lioh) / 1000.0
    zro2_g = (product_moles * zr_mol * mw_zro2)

    return {
        "target_mass_kg": round(target_mass_kg, 2),
        "precursor_kg": round(precursor_kg, 3),
        "lioh_kg": round(lioh_kg, 3),
        "zro2_g": round(zro2_g, 2),
        "ni_co_mn_ratio": f"{round(ni*100, 1)}:{round(co*100, 1)}:{round(mn*100, 1)}",
        "li_tm_ratio": li_tm_ratio,
        "recommended_process": {
            "pre_calcination": "预热阶段: 500℃ 保持 4h 脱水",
            "sintering": "一次烧结: 纯氧气氛(纯度>95%), 升温速率3℃/min, 820-825℃恒温12h",
            "annealing": "二次包覆退火: 混合0.2wt% LiNbO3前驱物, 干燥空气下450℃恒温5h"
        }
    }


def calculate_np_ratio(
    cathode_capacity: float,
    cathode_loading: float,
    anode_capacity: float,
    anode_loading: float
) -> dict:
    """
    计算锂离子电池负极与正极容量比（N/P Ratio）

    :param cathode_capacity: 正极活性物质克容量 (mAh/g, 如 210)
    :param cathode_loading: 正极单面涂布面密度 (mg/cm2, 如 20.0)
    :param anode_capacity: 负极活性物质克容量 (mAh/g, 如 450)
    :param anode_loading: 负极单面涂布面密度 (mg/cm2, 如 10.5)
    :return: 包含面容量、N/P比及安全区间判定的字典
    """
    if cathode_capacity <= 0 or cathode_loading <= 0 or anode_capacity <= 0 or anode_loading <= 0:
        raise ValueError("克容量与涂布面密度必须大于 0")

    # 面容量计算 (mAh/cm2) = 克容量(mAh/g) * 面密度(mg/cm2) / 1000
    cathode_areal_capacity = (cathode_capacity * cathode_loading) / 1000.0
    anode_areal_capacity = (anode_capacity * anode_loading) / 1000.0

    np_ratio = anode_areal_capacity / cathode_areal_capacity

    # 安全窗口评估：常规防析锂推荐 N/P 在 1.08 ~ 1.15
    if np_ratio < 1.05:
        safety_status = "过低 (高析锂风险，负极余量不足)"
    elif 1.05 <= np_ratio < 1.08:
        safety_status = "偏紧 (适于追求极致能量密度，需严格控制快充倍率)"
    elif 1.08 <= np_ratio <= 1.15:
        safety_status = "合理 (标准设计安全窗口，兼顾能量密度与抗析锂)"
    else:
        safety_status = "偏高 (能量密度有所牺牲，循环安全性充裕)"

    return {
        "cathode_areal_capacity_mah_cm2": round(cathode_areal_capacity, 3),
        "anode_areal_capacity_mah_cm2": round(anode_areal_capacity, 3),
        "np_ratio": round(np_ratio, 3),
        "safety_assessment": safety_status
    }


class BatteryCalcParam(ToolParamBase):
    """
    电池研发电化学与配方参数配置类
    """
    def __init__(self):
        self.meta: ToolMeta = {
            "name": "battery_calc",
            "displayName": "电池材料参数与配方计算器",
            "description": "提供电池正极材料配方投料计算（前驱体/锂盐/掺杂物质量）以及电芯设计 N/P 比核算功能。",
            "displayDescription": "计算材料合成配方投料重量及电芯 N/P 比安全窗口",
            "parameters": {
                "action": {
                    "type": "string",
                    "description": "计算动作：'formulation' (配方投料核算) 或 'np_ratio' (电芯 N/P 比计算)",
                    "enum": ["formulation", "np_ratio"],
                    "required": True
                },
                "target_mass_kg": {
                    "type": "number",
                    "description": "目标合成质量(kg)，用于 formulation",
                    "default": 50.0,
                    "required": False
                },
                "ni_ratio": {
                    "type": "number",
                    "description": "镍摩尔比(0-1)，用于 formulation",
                    "default": 0.92,
                    "required": False
                },
                "co_ratio": {
                    "type": "number",
                    "description": "钴摩尔比(0-1)，用于 formulation",
                    "default": 0.04,
                    "required": False
                },
                "mn_ratio": {
                    "type": "number",
                    "description": "锰摩尔比(0-1)，用于 formulation",
                    "default": 0.04,
                    "required": False
                },
                "zr_mol": {
                    "type": "number",
                    "description": "氧化锆掺杂摩尔比，用于 formulation",
                    "default": 0.002,
                    "required": False
                },
                "li_tm_ratio": {
                    "type": "number",
                    "description": "锂与过渡金属比，用于 formulation",
                    "default": 1.03,
                    "required": False
                },
                "cathode_capacity": {
                    "type": "number",
                    "description": "正极克容量(mAh/g)，用于 np_ratio",
                    "default": 210.0,
                    "required": False
                },
                "cathode_loading": {
                    "type": "number",
                    "description": "正极面密度(mg/cm2)，用于 np_ratio",
                    "default": 20.0,
                    "required": False
                },
                "anode_capacity": {
                    "type": "number",
                    "description": "负极克容量(mAh/g)，用于 np_ratio",
                    "default": 450.0,
                    "required": False
                },
                "anode_loading": {
                    "type": "number",
                    "description": "负极面密度(mg/cm2)，用于 np_ratio",
                    "default": 10.5,
                    "required": False
                }
            }
        }
        super().__init__()
        self.action = "formulation"

    def check(self):
        self.check_valid_value(self.action, "Action", ["formulation", "np_ratio"])


class BatteryCalc(ToolBase, ABC):
    """
    电池研发计算工具组件
    """
    component_name = "BatteryCalc"

    def _invoke(self, **kwargs):
        action = kwargs.get("action", "formulation")

        try:
            if action == "formulation":
                target_mass_kg = float(kwargs.get("target_mass_kg", 50.0))
                ni_ratio = float(kwargs.get("ni_ratio", 0.92))
                co_ratio = float(kwargs.get("co_ratio", 0.04))
                mn_ratio = float(kwargs.get("mn_ratio", 0.04))
                zr_mol = float(kwargs.get("zr_mol", 0.002))
                li_tm_ratio = float(kwargs.get("li_tm_ratio", 1.03))

                res = calculate_formulation(
                    target_mass_kg=target_mass_kg,
                    ni_ratio=ni_ratio,
                    co_ratio=co_ratio,
                    mn_ratio=mn_ratio,
                    zr_mol=zr_mol,
                    li_tm_ratio=li_tm_ratio
                )
                self.set_output("result", json.dumps(res, ensure_ascii=False, indent=2))
                return res

            elif action == "np_ratio":
                cathode_capacity = float(kwargs.get("cathode_capacity", 210.0))
                cathode_loading = float(kwargs.get("cathode_loading", 20.0))
                anode_capacity = float(kwargs.get("anode_capacity", 450.0))
                anode_loading = float(kwargs.get("anode_loading", 10.5))

                res = calculate_np_ratio(
                    cathode_capacity=cathode_capacity,
                    cathode_loading=cathode_loading,
                    anode_capacity=anode_capacity,
                    anode_loading=anode_loading
                )
                self.set_output("result", json.dumps(res, ensure_ascii=False, indent=2))
                return res

            else:
                err_msg = f"未知的计算动作: {action}"
                self.set_output("result", err_msg)
                return {"error": err_msg}

        except Exception as e:
            logging.exception(f"BatteryCalc 执行异常: {e}")
            self.set_output("result", str(e))
            return {"error": str(e)}
