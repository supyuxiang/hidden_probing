

class Language:
    def __init__(
        self,
        language_type:str,
    ):
        self.language_type = language_type
        self.system_prompt = 'You are a math expert.'
        self.load()
    
    def load(self):
        if self.language_type == 'en':
            self.user_prompt =(
                'CRITICAL RULE: All outputs MUST be written exclusively in English.'
                'Please solve the given math problem step by step and present the answer in the following format: "\\boxed{{X}}", where X is the answer.'
                '\n\nQuestion:\n{question}'
            )
            
        elif self.language_type == 'zh':
            self.user_prompt = (
                '关键规则：所有输出必须仅使用中文。'
                '请逐步求解给定的数学问题，并按以下格式呈现答案："\\boxed{{X}}"，其中X为答案。'
                '\n\n问题：\n{question}'
            )
    
        elif self.language_type == 'es':
            self.user_prompt = (
                'REGLA CRÍTICA: Toda la salida debe redactarse exclusivamente en español.'
                'Resuelve paso a paso el siguiente problema matemático y presenta la respuesta en el formato: "\\boxed{{X}}", donde X es la respuesta.'
                '\n\nPregunta:\n{question}'
            )
        elif self.language_type == 'vi':
            self.user_prompt = (
                'QUY TẮC NGHIÊM NGẶT: Tất cả đầu ra PHẢI được viết hoàn toàn bằng tiếng Việt.'
                'Vui lòng giải bài toán toán học đã cho từng bước một và trình bày đáp án theo định dạng sau: "\\boxed{{X}}", trong đó X là đáp án.'
                '\n\nCâu hỏi:\n{question}'
            )
        elif self.language_type == 'tr':
            self.user_prompt = (
                'KRİTİK KURAL: Tüm çıktılar KESİNLİKLE sadece Türkçe yazılmalıdır.'
                'Lütfen verilen matematik sorununu adım adım çözün ve cevabı şu formatta sunun: "\\boxed{{X}}", burada X cevaptır.'
                '\n\nSoru:\n{question}'
            )
        else:
            raise NotImplementedError
    