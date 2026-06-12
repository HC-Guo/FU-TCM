#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中医古籍XML转QA对生成工具
支持转接站API（OpenAI兼容格式）
"""

import json
import re
import time
import os
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# 禁用系统代理，避免VPN干扰
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ==================== API配置区域（用户自行填写）====================

API_KEY = os.getenv("TCM_API_KEY", "").strip()
API_URL = os.getenv("TCM_API_URL") or "https://api.minimaxi.com/v1/chat/completions"
MODEL_NAME = os.getenv("TCM_MODEL_NAME") or "MiniMax-M2.5"
# 可选配置
MAX_TOKENS = 2048
TEMPERATURE = 0.3  # 较低温度保证答案忠实于原文
CONCURRENT_REQUESTS = 50  # 并发数（根据API限制调整）
DELAY_BETWEEN_REQUESTS = 0.5  # 请求间隔（秒）

# ==================== 配置结束 ====================


def parse_xml_pian(content: str) -> List[Dict[str, str]]:
    """
    解析XML格式的古籍，按<篇名>切分
    返回: [{"pian_name": "篇名", "content": "内容"}, ...]
    """
    # 统一换行符：处理 \r\r\n 等各种变体
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    # 合并多个连续换行为两个换行
    content = re.sub(r'\n{3,}', '\n\n', content)

    # 匹配 <篇名>标题 + 内容/属性 的模式
    pattern = r'<篇名>(.+?)\n\n(?:内容|属性)：(.+?)(?=\n\n<目录>|\n\n<篇名>|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)

    pian_list = []
    for name, text in matches:
        # 清理内容中的多余换行
        text = re.sub(r'\n+', '\n', text).strip()
        pian_list.append({
            "pian_name": name.strip(),
            "content": text
        })

    return pian_list


def generate_qa_prompt(pian_name: str, content: str, book_name: str) -> str:
    """生成调用大模型的Prompt"""
    return f"""你是中医古籍研究专家。请根据《{book_name}》原文，针对「{pian_name}」生成多个问答对。

【原文】
{content}

【要求】
1. 识别内容是中药、药方、医理、医案或是其他，并根据内容对应维度生成问答对
2. 参考以下维度生成问答，只生成原文有的内容：
   - 中药：性味、功效主治、久服效果、产地（生长环境）、别名、各家论述、配伍禁忌、炮制用法等
   - 药方：组成药物（君臣佐使）、功效主治、制作方法、服用方法及时间、禁忌
   - 医理：根据文本内容生成问答对
   - 医案：如果有辨证过程（四诊-八纲-脏腑-证型-立法-选方-加减），按此推理过程生成问答对
   - 其他：根据文本内容生成问答对
3. 如果对于某个性质有不同的记载，放在一个问答对里
4. 如果有显然不对违背常识的内容，请剔除
5. 答案必须严格基于原文，不要添加原文外的知识；涵盖原文所有知识点

【★★★ 问题命名规则（必须严格遵守）★★★】
所有问题中禁止使用'这个''该''本方''此方'等代词，必须用具体名称指代。
根据篇名类型选择不同的命名策略：

■ 情况A：篇名是具体独特的方剂名（如'至宝丹''六味地黄丸'）
  → 直接用篇名作主语
  正确示例：至宝丹由哪些药物组成？/ 至宝丹的服用方法是什么？

■ 情况B：篇名是通用分类名（如'痢疾通治方''泄泻通治方''伤寒通治方'）
  → 必须在问题中加入原文中的【主治症状+核心药物】来区分
  正确示例：主治泄痢、含地榆酸石榴皮赤芍药的痢疾通治方由哪些药物组成？
  正确示例：主治泄痢不止、含黄连干姜的痢疾通治方的服用方法是什么？
  错误示例：痢疾通治方由哪些药物组成？（缺少区分标识，同名方剂无法区分）
  错误示例：痢疾通治方的主治功效是什么？（同上）

当前篇名「{pian_name}」属于哪种情况，请自行判断并严格执行对应规则。

【输出格式】
每对问答前标注类型，格式如下：

【类型】中药/药方/医理/医案/其他（根据内容选择其一）
Q1: [问题]
A1: [答案]

【类型】中药/药方/医理/医案/其他
Q2: [问题]
A2: [答案]

【类型】中药/药方/医理/医案/其他
Q3: [问题]
A3: [答案]"""

def call_llm_api(prompt: str) -> Optional[str]:
    """
    调用大模型API（OpenAI兼容格式）
    返回生成的文本，失败返回None
    """
    try:
        import requests

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "你是中医古籍专家，擅长从古籍原文中提炼知识并生成问答对。"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE
        }

        response = requests.post(API_URL, headers=headers, json=payload, timeout=300)
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    except Exception as e:
        print(f"    [错误] API调用失败: {e}")
        return None


def parse_qa_response(response: str) -> List[Dict[str, str]]:
    """
    解析大模型返回的文本，提取QA对和类型
    """
    qa_pairs = []

    # 按【类型】分割响应文本
    # 格式：【类型】类型名\nQ1:...\nA1:...
    parts = re.split(r'【类型】', response)

    for part in parts[1:]:  # 跳过第一个空部分
        if not part.strip():
            continue

        # 提取类型（第一行）
        lines = part.strip().split('\n', 1)
        qa_type = lines[0].strip() if lines else "其他"
        content = lines[1] if len(lines) > 1 else part

        # 在内容中提取Q和A
        q_pattern = r'[Qq]\d*[:：]\s*(.+?)(?=[Aa]\d*[:：]|$)'
        a_pattern = r'[Aa]\d*[:：]\s*(.+?)(?=[Qq]\d*[:：]|\Z)'

        q_match = re.search(q_pattern, content, re.DOTALL)
        a_match = re.search(a_pattern, content, re.DOTALL)

        if q_match and a_match:
            qa_pairs.append({
                "type": qa_type,
                "q": q_match.group(1).strip(),
                "a": a_match.group(1).strip()
            })

    # 如果没有匹配到类型格式，使用原始方式解析
    if not qa_pairs:
        pattern_q = r'[Qq]\d*[:：]\s*(.+?)(?=[Aa]\d*[:：]|$)'
        pattern_a = r'[Aa]\d*[:：]\s*(.+?)(?=[Qq]\d*[:：]|\Z)'
        questions = re.findall(pattern_q, response, re.DOTALL)
        answers = re.findall(pattern_a, response, re.DOTALL)

        for q, a in zip(questions, answers):
            qa_pairs.append({
                "type": "其他",
                "q": q.strip(),
                "a": a.strip()
            })

    return qa_pairs


def process_single_pian(pian: Dict, book_name: str) -> Dict:
    """处理单篇，返回结果字典（供并发调用）"""
    prompt = generate_qa_prompt(pian['pian_name'], pian['content'], book_name)

    max_retry = 3
    response = None
    for retry in range(max_retry):
        response = call_llm_api(prompt)
        if response:
            break
        time.sleep(2)

    if response:
        qa_pairs = parse_qa_response(response)
        return {
            "pian_id": pian['pian_id'],
            "pian_name": pian['pian_name'],
            "original_content": pian['content'][:500],
            "full_response": response,
            "qa_pairs": qa_pairs
        }
    else:
        return {
            "pian_id": pian['pian_id'],
            "pian_name": pian['pian_name'],
            "original_content": pian['content'][:500],
            "full_response": None,
            "qa_pairs": [],
            "error": "API调用失败"
        }


def process_single_book(input_file: Path, output_file: Path, checkpoint_file: Path = None):
    """处理单个古籍文件（支持断点续传 + 并发）"""

    # 提取书名
    book_name_map = {
        "000.txt": "神农本草经",
        "001.txt": "吴普本草",
    }
    book_name = book_name_map.get(input_file.name, input_file.stem)

    print(f"\n{'='*60}")
    print(f"处理书籍: {book_name}")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"{'='*60}\n")

    # 读取并解析
    print("[1/4] 读取并解析XML...")
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    pian_list = parse_xml_pian(content)
    for idx, p in enumerate(pian_list):
        p['pian_id'] = f"{idx:04d}_{p['pian_name']}"
    print(f"      共解析出 {len(pian_list)} 篇")

    # 检查断点续传
    successful_ids = set()
    failed_ids = set()
    results = None

    resume_file = None
    if output_file.exists():
        resume_file = output_file
    elif checkpoint_file and checkpoint_file.exists():
        resume_file = checkpoint_file
        print(f"      [断点] 从旧输出文件读取进度: {checkpoint_file}")

    if resume_file:
        print(f"\n[检测] 发现已有输出文件，尝试断点续传...")
        try:
            with open(resume_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)

            for item in existing_data.get('qa_data', []):
                item_id = item.get('pian_id', item.get('pian_name'))
                if item.get('qa_pairs') and len(item.get('qa_pairs', [])) > 0:
                    successful_ids.add(item_id)
                elif item.get('error'):
                    failed_ids.add(item_id)

            results = existing_data
            results['qa_data'] = [d for d in results['qa_data']
                                  if d.get('pian_id', d.get('pian_name')) in successful_ids]

            print(f"      已处理 {len(successful_ids) + len(failed_ids)} 篇（成功:{len(successful_ids)} 失败:{len(failed_ids)}）")
            print(f"      失败篇将重试")
        except Exception as e:
            print(f"      [警告] 读取已有文件失败: {e}，将重新处理")
            successful_ids = set()
            failed_ids = set()
            results = None

    # 旧格式兼容
    if successful_ids or failed_ids:
        old_ids = successful_ids | failed_ids
        has_new_format = any(id.startswith(('0','1','2','3','4','5','6','7','8','9')) and '_' in id for id in old_ids)
        if not has_new_format:
            name_to_ids = {}
            for p in pian_list:
                name_to_ids.setdefault(p['pian_name'], []).append(p['pian_id'])
            name_consume_idx = {}
            new_successful = set()
            for item in (results or {}).get('qa_data', []):
                pname = item.get('pian_name')
                if pname in name_to_ids:
                    ci = name_consume_idx.get(pname, 0)
                    if ci < len(name_to_ids[pname]):
                        pid = name_to_ids[pname][ci]
                        item['pian_id'] = pid
                        new_successful.add(pid)
                        name_consume_idx[pname] = ci + 1
            successful_ids = new_successful
            failed_ids = set()

    if results is None:
        results = {
            "book_name": book_name,
            "source_file": str(input_file),
            "total_pian": len(pian_list),
            "qa_data": []
        }

    # 清理旧数据：只保留有pian_id的条目，避免旧格式残留导致重复
    results['qa_data'] = [d for d in results['qa_data'] if d.get('pian_id')]

    # 筛选待处理篇目
    failed_pian_list = [p for p in pian_list if p['pian_id'] in failed_ids]
    new_pian_list = [p for p in pian_list if p['pian_id'] not in successful_ids and p['pian_id'] not in failed_ids]
    remaining_pian_list = failed_pian_list + new_pian_list

    if not remaining_pian_list:
        print("[完成] 所有篇目已处理完毕，无需继续")
        return

    print(f"[2/4] 开始并发调用API生成QA对（剩余{len(remaining_pian_list)}篇/{len(pian_list)}篇，并发数{CONCURRENT_REQUESTS}）...")

    completed_count = 0
    with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
        future_to_pian = {
            executor.submit(process_single_pian, pian, book_name): pian
            for pian in remaining_pian_list
        }

        for future in as_completed(future_to_pian):
            result = future.result()
            results["qa_data"].append(result)
            completed_count += 1

            if result.get('qa_pairs'):
                print(f"  [{completed_count}/{len(remaining_pian_list)}] {result['pian_name'][:30]} - {len(result['qa_pairs'])}个QA对")
            else:
                print(f"  [{completed_count}/{len(remaining_pian_list)}] {result['pian_name'][:30]} - 失败")

            if completed_count % 50 == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"    [保存] 已保存中间结果（{completed_count}/{len(remaining_pian_list)}篇）")

    # 按pian_id去重（后出现的覆盖先出现的，即新结果覆盖旧数据）
    dedup = {}
    for d in results['qa_data']:
        pid = d.get('pian_id')
        if pid:
            dedup[pid] = d
    results['qa_data'] = list(dedup.values())

    # 最终保存
    print(f"\n[3/4] 保存最终结果...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"      结果已保存至: {output_file}")

    # 统计
    success_count = sum(1 for d in results['qa_data'] if d.get('qa_pairs'))
    fail_count = sum(1 for d in results['qa_data'] if d.get('error'))
    total_qa = sum(len(d['qa_pairs']) for d in results['qa_data'])
    print(f"\n{'='*60}")
    print(f"处理完成!")
    print(f"  - 成功: {success_count} 篇")
    print(f"  - 失败: {fail_count} 篇")
    print(f"  - 生成QA对总数: {total_qa} 个")
    print(f"  - 结果保存至: {output_file}")
    print(f"{'='*60}")




def main():
    """主函数 - 批量处理04_clinical/standard目录下所有文件"""

    # 目录配置
    INPUT_DIR = PROJECT_ROOT / "中医古籍分类/04_clinical/standard"
    CHECKPOINT_DIR = PROJECT_ROOT / "qa_output/04_clinical"       # 旧输出，用于读断点
    OUTPUT_DIR = PROJECT_ROOT / "qa_output_v2/04_clinical"        # 新输出

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 验证API配置
    if not API_KEY:
        print("[警告] API_KEY未配置，请先设置环境变量 TCM_API_KEY 后再运行。")
        return

    # 获取所有txt文件并排序
    txt_files = sorted(INPUT_DIR.glob("*.txt"))
    if not txt_files:
        print(f"[错误] 目录 {INPUT_DIR} 下没有找到.txt文件")
        return

    print(f"\n{'='*60}")
    print(f"批量处理模式")
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"共发现 {len(txt_files)} 个文件")
    print(f"{'='*60}\n")

    # 批量处理
    total_start_time = time.time()
    for i, input_file in enumerate(txt_files, 1):
        # 输出文件名保持序号一致，如 049.txt -> 049_qa.json
        output_file = OUTPUT_DIR / f"{input_file.stem}_qa.json"
        checkpoint_file = CHECKPOINT_DIR / f"{input_file.stem}_qa.json"

        print(f"\n{'#'*60}")
        print(f"[{i}/{len(txt_files)}] 开始处理: {input_file.name}")
        print(f"{'#'*60}")

        process_single_book(input_file, output_file, checkpoint_file)

        # 文件间延迟，避免API限流
        if i < len(txt_files):
            print(f"\n[延迟] 等待 {DELAY_BETWEEN_REQUESTS} 秒后继续下一个文件...")
            time.sleep(DELAY_BETWEEN_REQUESTS)

    total_elapsed = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"全部处理完成!")
    print(f"  - 总文件数: {len(txt_files)}")
    print(f"  - 总耗时: {total_elapsed:.1f} 秒 ({total_elapsed/60:.1f} 分钟)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
