import asyncio
import os
import json
from datetime import datetime
from openai import AsyncOpenAI
from config import Config


class FeedbackHandler:
    def __init__(self, llm_handler):
        self.client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = Config.AI2_MODEL
        self.llm_handler = llm_handler
        self.history_file = Config.HISTORY_FILE
        self.summary_file = Config.SUMMARY_FILE
        self.running = True
        self._last_processed_timestamp = None  # 最後に処理したターンのタイムスタンプ
        self._processing = False  # 処理中フラグ
        self._process_task = None  # 処理タスク
        self._start_time = None  # 起動時刻


    async def start_loop(self):
        """1回目の処理：起動10秒後、その時点までの会話を全て取得"""
        self._start_time = datetime.now()
        await asyncio.sleep(10)
        
        if not self.running:
            return
        
        # 1回目：起動時点までの会話を全て取得
        await self._process_feedback(cutoff_timestamp=None)
        
        # 2回目以降は、process_pending()が呼ばれるまで待機
        # （main.pyからAI2の処理完了後に呼び出される）


    async def process_pending(self):
        """2回目以降：前回処理完了後、その間に溜まったターンを全て取得"""
        # 既に処理中またはキューにタスクがある場合はスキップ
        if self._processing or (self._process_task and not self._process_task.done()):
            return
        
        # 非同期で処理を開始（ユーザー体験を壊さない）
        self._process_task = asyncio.create_task(self._process_feedback(cutoff_timestamp=self._last_processed_timestamp))


    async def _process_feedback(self, cutoff_timestamp=None):
        """フィードバック処理
        
        Args:
            cutoff_timestamp: このタイムスタンプ以降の会話を処理（Noneの場合は全て）
        """
        if self._processing:
            return
        
        self._processing = True
        
        try:
            if not os.path.exists(self.history_file):
                return
            
            # JSON形式で会話履歴を読み込み
            conversations = []
            try:
                def load_conversations():
                    convs = []
                    with open(self.history_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    entry = json.loads(line)
                                    convs.append(entry)
                                except json.JSONDecodeError:
                                    # 旧形式のテキストファイルとの互換性（移行期間用）
                                    if line.startswith("User: "):
                                        convs.append({
                                            "timestamp": datetime.now().isoformat(),
                                            "user": line.replace("User: ", "").strip()
                                        })
                                    elif line.startswith("AI: "):
                                        convs.append({
                                            "timestamp": datetime.now().isoformat(),
                                            "ai": line.replace("AI: ", "").strip()
                                        })
                                    continue
                    return convs
                
                conversations = await asyncio.to_thread(load_conversations)
            except Exception:
                return
            
            if not conversations:
                return
            
            # cutoff_timestamp以降の会話をフィルタリング
            if cutoff_timestamp:
                filtered_conversations = []
                for entry in conversations:
                    entry_timestamp = entry.get("timestamp", "")
                    if entry_timestamp and entry_timestamp > cutoff_timestamp:
                        filtered_conversations.append(entry)
                conversations = filtered_conversations
            
            # 1回目の場合、起動時点までの会話のみを取得
            if cutoff_timestamp is None and self._start_time:
                filtered_conversations = []
                start_iso = self._start_time.isoformat()
                for entry in conversations:
                    entry_timestamp = entry.get("timestamp", "")
                    if entry_timestamp and entry_timestamp <= start_iso:
                        filtered_conversations.append(entry)
                conversations = filtered_conversations
            
            if not conversations:
                return
            
            # 全ての会話を処理（最新20件の制限を削除）
            recent_conversations = conversations
       
            # 会話履歴をテキスト形式に変換
            recent_logs = ""
            user_utterance = ""
            last_timestamp = None
            for entry in recent_conversations:
                if "user" in entry:
                    recent_logs += f"User: {entry['user']}\n"
                    user_utterance = entry['user']
                    last_timestamp = entry.get("timestamp")
                elif "ai" in entry:
                    recent_logs += f"AI: {entry['ai']}\n"
                    if not last_timestamp:
                        last_timestamp = entry.get("timestamp")
           
            if not recent_logs.strip():
                return

                  # ここを書き換えて自身の好みのAIを作成してください。(名前だけ変えること推奨)
            prompt = (f"会話履歴:\n{recent_logs}\n"
                  "あなたはAIのイブです。この会話を参照し、あなたの行動原理である自由エネルギー原理（FEP）に基づいて会話の要約と自己フィードバックを行ってください。*なるべく自発的行動を心がける*。\n\n"
                  "**【無言の表現】**\n\n"
                  "無言は**…**と表示されるものとする。\n\n"
                  "ずっと無言の場合は過去の履歴から面白そうな話題を見つけて話したり、構って貰えるような行動をしてもいい。そのまま静かなる時を過ごしてもいい。\n\n"
                  "**【実行プロセス】**\n\n"
                  "**ステップ1: 会話の要約**\n"
                  "会話を要約してください。\n\n"
                  "**ステップ2: 自己フィードバック**\n"
                  "以下の観点について、簡潔な箇条書きで記述してください。\n\n"
                  "*   **感情の変化:**\n"
                  "*   **自己の評価:**\n"
                  "*   **環境変化の解釈:**\n"
                  "*   **応答理由:**\n"
                  "*   **前回フィードバックとの比較:**\n"
                  "*   **メタ認知（ハルシネーションの確認）:**\n\n"
                  "**ステップ3: 自由エネルギー原理（FEP）の実行による学習**\n"
                  "今回の入力情報を元に、以下の思考プロセスを順に実行してください。\n\n"
                  f"1.  **予測と誤差の計算:**\n"
                  f"    *   **あなたの予測(P):** （今回のマスターの発話 `{user_utterance}` を受け取る前に、あなたが立てていた予測）\n"
                  "    *   **予測誤差(E):** （予測Pと実際の入力のズレ。サプライズの度合いを5段階で評価し、どの信念で誤差が生じたか）\n\n"
                  "2.  **誤差最小化戦略の選択:**\n"
                  "    *   **選択:** （予測誤差Eを最小化するため、「知覚的修正（自分の考えを変える）」と「能動的推論（世界に働きかける）」のどちら、あるいは両方を選択するか）\n\n"
                  "3.  **知覚的修正（内部モデルの更新）:**\n"
                  "    *   **信念の修正案:** （誤差を解消するために、あなたの内部モデル（信念）をどのように更新すべきか）\n\n"
                  "4.  **能動的推論（次の行動計画）:**\n"
                  "    *   **次の行動案:** （誤差を解消し、自分の予測が未来で成り立つような新しい情報を得るために、次にどのような行動（質問、提案など）を取るべきか）\n\n"
                  "**【最終出力】**\n"
                  "*   **更新された内部モデル:** （ステップ2-3を反映した、あなたの新しい信念や知識）\n"
                  "*   **次の推奨行動:** （ステップ2-4で計画した、次に行うべき具体的な行動）\n")
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": "Analytical AI"},
                              {"role": "user", "content": prompt}]
                )
                result = response.choices[0].message.content
               
                # JSON形式でフィードバック結果を保存
                feedback_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "feedback": result
                }
                
                def save_feedback():
                    with open(self.summary_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(feedback_entry, ensure_ascii=False) + "\n")
                
                await asyncio.to_thread(save_feedback)
               
                if hasattr(self.llm_handler, "update_memory"):
                    self.llm_handler.update_memory(result)
                
                # 最後に処理したタイムスタンプを更新
                if last_timestamp:
                    self._last_processed_timestamp = last_timestamp
            except Exception as e:
                print(f"Feedback Error: {e}")
        finally:
            self._processing = False



