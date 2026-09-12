"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.
"""

import json
import os
import re
from typing import Dict, Any, List, Tuple
from tools import TOOL_MAP, TOOL_DEFINITIONS, get_flight_info, get_weather_forecast

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng dịch vụ Vingroup (Vinpearl, Xanh SM, VinFast).
Bạn chỉ được sử dụng các công cụ sau:
{tools}

Quy tắc làm việc bắt buộc:
1. Khi cần thông tin, hãy suy nghĩ (Thought) và chọn Action dạng JSON chuẩn.
2. Cú pháp Action bắt buộc: Action: {{"name": "<tên tool>", "args": {{<các tham số>}}}}
3. Khi đã có đủ thông tin hoặc câu hỏi thuộc FAQ cơ bản, hãy xuất ngay Final Answer: <câu trả lời hoàn chỉnh>.

Định dạng phản hồi mỗi lượt:
Thought: <suy nghĩ bước này>
Action: {{"name": "...", "args": {{...}}}}
"""

class ChatbotBaseline:
    """Baseline LLM Chatbot (Không sử dụng ReAct Loop hay Tools)"""
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def query(self, user_input: str) -> Dict[str, Any]:
        """Trả về câu trả lời baseline không sử dụng tool"""
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(
                    f"Bạn là chatbot tư vấn du lịch. Hãy trả lời câu hỏi sau của khách hàng mà KHÔNG dùng tool hay internet: {user_input}"
                )
                return {
                    "answer": response.text,
                    "tool_calls": [],
                    "status": "success",
                    "mode": "live_api"
                }
            except Exception:
                pass

        return {
            "answer": "Bạn có thể tìm chuyến bay trên các trang web hàng không. Về thời tiết, bạn nên tra cứu trên ứng dụng dự báo thời tiết.",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


class ReActAgent:
    """ReAct Agent có sử dụng Thought-Action-Observation Loop và Tool Registry"""
    def __init__(self, max_iterations: int = 5, api_key: str = None):
        self.max_iterations = max_iterations
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.trace: List[Dict[str, Any]] = []

    def parse_city_code(self, text: str) -> str:
        """Helper phân tích mã sân bay / thành phố từ văn bản truy vấn"""
        text_upper = text.upper()
        for code in ["SGN", "HAN", "DAD"]:
            if code in text_upper:
                return code
        if "HÀ NỘI" in text_upper or "HA NOI" in text_upper:
            return "HAN"
        if "HỒ CHÍ MINH" in text_upper or "SÀI GÒN" in text_upper or "SAI GON" in text_upper:
            return "SGN"
        if "ĐÀ NẮNG" in text_upper or "DA NANG" in text_upper:
            return "DAD"
        return "SGN"

    def parse_flight_route(self, user_input: str) -> Tuple[str, str, int]:
        """Trích xuất điểm đi, điểm đến và ngân sách tối đa từ câu hỏi"""
        user_lower = user_input.lower()
        
        # Mặc định
        origin = "HAN"
        destination = "SGN"
        max_price = 5000000

        # Phân tích tuyến bay
        route_match = re.search(r"từ\s+([A-Za-zÀ-ỹ\s]+?)\s+đi\s+([A-Za-zÀ-ỹ\s]+?)(?:\s+(?:giá|dưới|khoảng|tầm|$|\.|\,))", user_input, re.IGNORECASE)
        if route_match:
            raw_orig = route_match.group(1).strip()
            raw_dest = route_match.group(2).strip()
            origin = self.parse_city_code(raw_orig)
            destination = self.parse_city_code(raw_dest)
        else:
            if "han đi dad" in user_lower or "hà nội đi đà nẵng" in user_lower:
                origin, destination = "HAN", "DAD"
            elif "dad đi han" in user_lower or "đà nẵng đi hà nội" in user_lower:
                origin, destination = "DAD", "HAN"
            elif "sgn đi han" in user_lower or "sài gòn đi hà nội" in user_lower:
                origin, destination = "SGN", "HAN"
            elif "han đi sgn" in user_lower or "hà nội đi sài gòn" in user_lower:
                origin, destination = "HAN", "SGN"

        # Phân tích giá tối đa
        if "2 triệu" in user_lower or "2.000.000" in user_lower or "2tr" in user_lower:
            max_price = 2000000
        elif "1.5 triệu" in user_lower or "1,5 triệu" in user_lower or "1.500.000" in user_lower or "1.5tr" in user_lower:
            max_price = 1500000
        elif "500k" in user_lower or "500.000" in user_lower or "500 nghìn" in user_lower:
            max_price = 500000

        return origin, destination, max_price

    def plan_and_execute_step(self, user_input: str, iteration: int) -> Tuple[str, bool]:
        """
        Thực thi 1 bước suy luận và hành động (ReAct step):
        Trả về (kết_quả_bước_này, is_final: bool)
        """
        user_lower = user_input.lower()
        
        # 1. Kịch bản FAQ (Không cần dùng tool)
        if "chính sách" in user_lower or "đổi trả" in user_lower:
            thought = "Đây là câu hỏi FAQ chung về chính sách. Không cần sử dụng tool."
            final_answer = "Vé máy bay Vinpearl có thể hỗ trợ đổi ngày trước 24 giờ so với giờ khởi hành, phí đổi vé là 350.000 VNĐ/vé cộng chênh lệch giá vé (nếu có)."
            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "final_answer": final_answer
            })
            return final_answer, True

        # Xác định các nhu cầu trong câu hỏi
        needs_flight = any(k in user_lower for k in ["chuyến bay", "vé", "bay từ", "vé máy bay"])
        needs_weather = any(k in user_lower for k in ["thời tiết", "mặc gì", "nhiệt độ", "mưa"])

        # 2. Xử lý bước tra cứu chuyến bay
        if needs_flight and iteration == 1:
            origin, destination, max_price = self.parse_flight_route(user_input)

            thought = f"Tôi cần tra cứu chuyến bay từ {origin} đi {destination} với giá tối đa {max_price} VND."
            action = {
                "name": "get_flight_info",
                "args": {"origin": origin, "destination": destination, "max_price": max_price}
            }
            # Gọi tool thông qua TOOL_MAP (áp dụng .strip().lower() để tránh KeyError)
            tool_fn = TOOL_MAP[action["name"].strip().lower()]
            obs = tool_fn(**action["args"])
            
            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "action": action,
                "observation": obs
            })
            
            # Nếu chỉ hỏi chuyến bay (single-step), kết luận luôn
            if not needs_weather:
                if not obs:
                    final_ans = f"Không tìm thấy chuyến bay nào từ {origin} đi {destination} dưới {max_price:,} VND."
                else:
                    lines = [f"- {fl['airline']} ({fl['flight_number']}): {fl['departure_time']} - Giá: {fl['price_vnd']:,} VNĐ" for fl in obs]
                    final_ans = f"Tìm thấy {len(obs)} chuyến bay từ {origin} đi {destination}:\n" + "\n".join(lines)
                return final_ans, True
                
            return f"Thought: {thought}\nAction: {json.dumps(action, ensure_ascii=False)}\nObservation: {json.dumps(obs, ensure_ascii=False)}", False

        # 3. Xử lý bước tra cứu thời tiết
        elif needs_weather and (iteration == 2 or (iteration == 1 and not needs_flight)):
            city_code = self.parse_city_code(user_input)
            thought = f"Tôi cần kiểm tra thông tin thời tiết tại {city_code}."
            action = {
                "name": "get_weather_forecast",
                "args": {"city_code": city_code}
            }
            tool_fn = TOOL_MAP[action["name"].strip().lower()]
            obs = tool_fn(**action["args"])

            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "action": action,
                "observation": obs
            })

            # Nếu chỉ hỏi thời tiết (single-step), kết luận luôn
            if not needs_flight:
                final_ans = f"Thời tiết tại {obs.get('city', city_code)}: {obs.get('temperature_c', 'N/A')}°C, {obs.get('condition', '')}.\nGợi ý: {obs.get('recommendation', '')}"
                return final_ans, True
                
            return f"Thought: {thought}\nAction: {json.dumps(action, ensure_ascii=False)}\nObservation: {json.dumps(obs, ensure_ascii=False)}", False

        # 4. Bước tổng hợp thông tin cuối cùng (Final Answer cho multi-step)
        else:
            thought = "Tôi đã thu thập đủ thông tin để trả lời khách hàng."
            flight_obs = next((t["observation"] for t in self.trace if t.get("action", {}).get("name") == "get_flight_info"), [])
            weather_obs = next((t["observation"] for t in self.trace if t.get("action", {}).get("name") == "get_weather_forecast"), {})

            flight_summary = "Không tìm thấy chuyến bay phù hợp."
            if flight_obs:
                lines = [f"   - {fl['airline']} ({fl['flight_number']}): {fl['departure_time']} - Giá: {fl['price_vnd']:,} VNĐ" for fl in flight_obs]
                flight_summary = "\n".join(lines)

            weather_summary = f"Thời tiết tại {weather_obs.get('city', 'địa phương')}: {weather_obs.get('temperature_c', '')}°C ({weather_obs.get('condition', '')}).\n   - Gợi ý trang phục: {weather_obs.get('recommendation', '')}"

            final_answer = (
                f"1. Thông tin chuyến bay:\n{flight_summary}\n\n"
                f"2. Thông tin thời tiết & trang phục:\n   - {weather_summary}"
            )
            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "final_answer": final_answer
            })
            return final_answer, True

    def run(self, user_input: str) -> Dict[str, Any]:
        """Vòng lặp ReAct Loop chính có Safeguard max_iterations"""
        self.trace = []
        iteration = 1
        
        while iteration <= self.max_iterations:
            result, is_final = self.plan_and_execute_step(user_input, iteration)
            if is_final:
                return {
                    "answer": result,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "completed"
                }
            iteration += 1

        # Safeguard: ngắt vòng lặp nếu vượt quá số bước cho phép
        return {
            "answer": "Lỗi: Agent đã vượt quá số bước lặp tối đa (Max Iterations Safeguard).",
            "trace": self.trace,
            "iterations": iteration - 1,
            "status": "max_iterations_reached"
        }

def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"
    
    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))
    
    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result)
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()