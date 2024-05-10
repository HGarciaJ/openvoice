import os
import torch
import argparse
import gradio as gr
from zipfile import ZipFile
import langid
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS

parser = argparse.ArgumentParser()
parser.add_argument("--share", action='store_true', default=False, help="make link public")
args = parser.parse_args()

# Definir las rutas y configuraciones
ckpt_converter = 'checkpoints_v2/converter'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = 'outputs_v2'
os.makedirs(output_dir, exist_ok=True)

# Cargar el conversor de color de tono
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

# Función para predecir con el modelo de síntesis de voz v2
def predict(prompt, style, audio_file_pth, agree):
    # Inicializar mensaje de texto vacío
    text_hint = ''
    # Comprobar si se aceptan los términos
    if not agree:
        text_hint += '[ERROR] ¡Por favor acepta los Términos y Condiciones!\n'
        return (
            text_hint,
            None,
            None,
        )

    # Obtener el tono de color del audio del altavoz de referencia
    reference_speaker = audio_file_pth
    target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=False)

    # Textos de ejemplo para diferentes idiomas
    texts = {
        'EN': "Did you ever hear a folk tale about a giant turtle?",
        'ES': "El resplandor del sol acaricia las olas, pintando el cielo con una paleta deslumbrante.",
        'FR': "La lueur dorée du soleil caresse les vagues, peignant le ciel d'une palette éblouissante.",
        'ZH': "在这次vacation中，我们计划去Paris欣赏埃菲尔铁塔和卢浮宫的美景。",
        'JP': "彼は毎朝ジョギングをして体を健康に保っています。",
        'KR': "안녕하세요! 오늘은 날씨가 정말 좋네요.",
    }

    # Velocidad ajustable
    speed = 1.0

    # Iterar sobre los textos de ejemplo para generar audio en diferentes idiomas
    for language, text in texts.items():
        model = TTS(language=language, device=device)
        speaker_ids = model.hps.data.spk2id
        
        for speaker_key in speaker_ids.keys():
            speaker_id = speaker_ids[speaker_key]
            speaker_key = speaker_key.lower().replace('_', '-')
            
            # Cargar el tono de color del altavoz de origen
            source_se = torch.load(f'checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)
            
            # Generar audio con el modelo TTS para el texto dado y el altavoz específico
            src_path = f'{output_dir}/tmp.wav'
            model.tts_to_file(text, speaker_id, src_path, speed=speed)
            
            # Aplicar el tono de color al audio generado
            save_path = f'{output_dir}/output_v2_{speaker_key}_{language}.wav'
            encode_message = "@MyShell"
            tone_color_converter.convert(
                audio_src_path=src_path, 
                src_se=source_se, 
                tgt_se=target_se, 
                output_path=save_path,
                message=encode_message)
    
    # Mensaje de éxito
    text_hint += f'''¡Respuesta generada con éxito!\n'''

    return (
        text_hint,
        save_path,  # Ruta del archivo de audio generado
        reference_speaker,  # Ruta del audio del altavoz de referencia
    )

title = "MyShell OpenVoice"

description = """
We introduce OpenVoice, a versatile instant voice cloning approach that requires only a short audio clip from the reference speaker to replicate their voice and generate speech in multiple languages. OpenVoice enables granular control over voice styles, including emotion, accent, rhythm, pauses, and intonation, in addition to replicating the tone color of the reference speaker. OpenVoice also achieves zero-shot cross-lingual voice cloning for languages not included in the massive-speaker training set.
"""

markdown_table = """
<div align="center" style="margin-bottom: 10px;">

|               |               |               |
| :-----------: | :-----------: | :-----------: | 
| **OpenSource Repo** | **Project Page** | **Join the Community** |        
| <div style='text-align: center;'><a style="display:inline-block,align:center" href='https://github.com/myshell-ai/OpenVoice'><img src='https://img.shields.io/github/stars/myshell-ai/OpenVoice?style=social' /></a></div> | [OpenVoice](https://research.myshell.ai/open-voice) | [![Discord](https://img.shields.io/discord/1122227993805336617?color=%239B59B6&label=%20Discord%20)](https://discord.gg/myshell) |

</div>
"""

markdown_table_v2 = """
<div align="center" style="margin-bottom: 2px;">

|               |               |               |              |
| :-----------: | :-----------: | :-----------: | :-----------: | 
| **OpenSource Repo** | <div style='text-align: center;'><a style="display:inline-block,align:center" href='https://github.com/myshell-ai/OpenVoice'><img src='https://img.shields.io/github/stars/myshell-ai/OpenVoice?style=social' /></a></div> |  **Project Page** |  [OpenVoice](https://research.myshell.ai/open-voice) |     

| | |
| :-----------: | :-----------: |
**Join the Community** |   [![Discord](https://img.shields.io/discord/1122227993805336617?color=%239B59B6&label=%20Discord%20)](https://discord.gg/myshell) |

</div>
"""
content = """
<div>
  <strong>If the generated voice does not sound like the reference voice, please refer to <a href='https://github.com/myshell-ai/OpenVoice/blob/main/docs/QA.md'>this QnA</a>.</strong> <strong>For multi-lingual & cross-lingual examples, please refer to <a href='https://github.com/myshell-ai/OpenVoice/blob/main/demo_part2.ipynb'>this jupyter notebook</a>.</strong>
  This online demo mainly supports <strong>English</strong>. The <em>default</em> style also supports <strong>Chinese</strong>. But OpenVoice can adapt to any other language as long as a base speaker is provided.
</div>
"""
wrapped_markdown_content = f"<div style='border: 1px solid #000; padding: 10px;'>{content}</div>"


examples = [
    [
        "今天天气真好，我们一起出去吃饭吧。",
        'default',
        "resources/demo_speaker1.mp3",
        True,
    ],[
        "This audio is generated by open voice with a half-performance model.",
        'whispering',
        "resources/demo_speaker2.mp3",
        True,
    ],
    [
        "He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered, flour-fattened sauce.",
        'sad',
        "resources/demo_speaker0.mp3",
        True,
    ],
]

examples = [
    [
        "今天天气真好，我们一起出去吃饭吧。",
        'default',
        "resources/demo_speaker1.mp3",
        True,
    ],[
        "This audio is generated by open voice with a half-performance model.",
        'whispering',
        "resources/demo_speaker2.mp3",
        True,
    ],
    [
        "He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered, flour-fattened sauce.",
        'sad',
        "resources/demo_speaker0.mp3",
        True,
    ],
]

# Definir la interfaz de Gradio
with gr.Blocks(analytics_enabled=False) as demo:
    input_text_gr = gr.Textbox(
        label="Text Prompt",
        info="One or two sentences at a time is better. Up to 200 text characters.",
        value="He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered, flour-fattened sauce.",
    )
    style_gr = gr.Dropdown(
        label="Style",
        info="Select a style of output audio for the synthesised speech. (Chinese only support 'default' now)",
        choices=['default', 'whispering', 'cheerful', 'terrified', 'angry', 'sad', 'friendly'],
        max_choices=1,
        value="default",
    )
    ref_gr = gr.Audio(
        label="Reference Audio",
        info="Click on the ✎ button to upload your own target speaker audio",
        type="filepath",
        value="resources/demo_speaker2.mp3",
    )
    tos_gr = gr.Checkbox(
        label="Agree",
        value=False,
        info="I agree to the terms of the cc-by-nc-4.0 license-: https://github.com/myshell-ai/OpenVoice/blob/main/LICENSE",
    )

    tts_button = gr.Button("Send", elem_id="send-btn", visible=True)

    out_text_gr = gr.Text(label="Info")
    audio_gr = gr.Audio(label="Synthesised Audio", autoplay=True)
    ref_audio_gr = gr.Audio(label="Reference Audio Used")

    gr.Examples(examples,
                label="Examples",
                inputs=[input_text_gr, style_gr, ref_gr, tos_gr],
                outputs=[out_text_gr, audio_gr, ref_audio_gr],
                fn=predict,
                cache_examples=False,)
    tts_button.click(predict, [input_text_gr, style_gr, ref_gr, tos_gr], outputs=[out_text_gr, audio_gr, ref_audio_gr])

demo.queue()  
demo.launch(debug=True, show_api=True, share=args.share)
