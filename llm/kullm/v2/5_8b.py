# pip install -U torch transformers tokenizers accelerate jupyter
# pip install prompter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from utils.prompter import Prompter

device = ("cuda" if torch.cuda.is_available() else "mps")

tokenizer = AutoTokenizer.from_pretrained("nlpai-lab/kullm-polyglot-5.8b-v2")
model = AutoModelForCausalLM.from_pretrained("nlpai-lab/kullm-polyglot-5.8b-v2").to(device=device)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
prompter = Prompter("kullm")


def infer(instruction="", input_text=""):
    prompt = prompter.generate_prompt(instruction, input_text)
    output = pipe(
        prompt, max_length=512,
        temperature=0.2,
        repetition_penalty=3.0,
        num_beams=5,
        eos_token_id=2
    )
    s = output[0]["generated_text"]
    result = prompter.get_response(s)

    return result


print(infer(input_text="오늘 서울 날씨"))
print(infer(input_text="""다음을 한국어로 번역해:
...
... We both worked with young moms who go completely dark for a few hours in the evening, when they are with their families and putting their kids to bed.
... """))

data = "삼성전자의 무선 이어폰 갤럭시 버즈에도 콩나물 줄기, 이른바 '스템(기둥)'이 생긴다. 내달 갤럭시 언팩에서 공개될 것으로 알려진 갤럭시 버즈3 시리즈의 패키지 이미지가 공개됐다. 18일(현지시간) IT매체 91모바일은 삼성 갤럭시 버즈3의 소매용 패키지 이미지를 공개했다. 해당 사진을 살피면 전작들과 달리 차세대 갤럭시 버즈는 스템 디자인이 확연하다. 인이어 모양이 달라짐에 따라, 이어버드 충전 및 휴대 케이스 역시 전작대비 넓어진 모습이다. 박스 뒷면의 상세 설명에 따르면 갤럭시 버즈3시리즈에는 충전 상태를 드러내는 LED 표시기가 탑재된다. 업계 전망에 따르면 해당 LED조명은 버즈3프로에 탑재되며, 이어버드 배터리 및 음악 볼륨 상태 등을 시각화해 줄 것으로 관측된다. 갤럭시 버즈3프로는 한 번 충전으로 최대 6시간, 일반 버즈3는 최대 5시간 사용 가능하다. 프로 모델에는 양방향 스피커에 ANC(노이즈캔슬링), 적응형 소음제어 및 주변 사운드 모드 기능이 탑재되나, 일반형은 해당 기능이 빠질 것으로 예상된다. 또한 이번 신작 시리즈에는 땀과 물을 조절할 수 있는 고정형 이어 후크 등도 제공될 예정이다. 반면, 전작 대비 다운그레이드도 있다. 이번 신작은 10분 급속 충전으로 1시간가량 사용할 수 있는 고속 충전도 지원되지만, 갤럭시 버즈2의 경우 5분 충전으로 1시간 재생을 제공했다. 한편, 최근 삼성전자가 자사 '갤럭시 AI'를 모바일 기기에 확장하고 있는만큼 갤럭시 버즈3시리즈 역시 AI가 적용될 예정이다. 신제품은 오는 7월 10일 프랑스 파리에서 열리는 갤럭시 언팩을 통해 공개된다."
result_by_data = infer(instruction=data, input_text="갤럭시 언팩에서 공개될 제품이 뭐야")
print(result_by_data)