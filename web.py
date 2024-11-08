# Open Source Model Licensed under the Apache License Version 2.0 and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software and/or models in this distribution may have been
# modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.l

import os
from PIL import Image
from flask import Flask, request, jsonify
from infer import Text2Image, Removebg, Image2Views, Views2Mesh, GifRenderer

app = Flask(__name__)

# Initialize models
rembg_model = Removebg()
image_to_views_model = None
views_to_mesh_model = None
text_to_image_model = None
gif_renderer = None


# 在项目启动时实例化模型
def initialize_models():
    global image_to_views_model, views_to_mesh_model, text_to_image_model, gif_renderer
    device = os.getenv('CUDA_DEVICE_ENV', "cuda:1")
    use_lite = False
    mv23d_cfg_path = './svrm/configs/svrm.yaml'
    mv23d_ckt_path = 'weights/svrm/svrm.safetensors'
    text2image_path = 'weights/hunyuanDiT'
    image_to_views_model = Image2Views(device=device, use_lite=use_lite)
    views_to_mesh_model = Views2Mesh(mv23d_cfg_path, mv23d_ckt_path, device, use_lite=use_lite)
    text_to_image_model = Text2Image(pretrain=text2image_path, device=device, save_memory=False)
    gif_renderer = GifRenderer(device=device)


initialize_models()


@app.route('/generate3d', methods=['POST'])
def generate_3d():
    data = request.json
    text_prompt = data.get('text_prompt', '')
    image_prompt = data.get('image_prompt', '')
    device = data.get('device', os.getenv('CUDA_DEVICE_ENV', "cuda:1"))
    use_lite = data.get('use_lite', False)
    mv23d_cfg_path = data.get('mv23d_cfg_path', './svrm/configs/svrm.yaml')
    mv23d_ckt_path = data.get('mv23d_ckt_path', 'weights/svrm/svrm.safetensors')
    text2image_path = data.get('text2image_path', 'weights/hunyuanDiT')
    save_folder = data.get('save_folder', './outputs/test/')
    t2i_seed = data.get('t2i_seed', 0)
    t2i_steps = data.get('t2i_steps', 25)
    gen_seed = data.get('gen_seed', 0)
    gen_steps = data.get('gen_steps', 50)
    max_faces_num = data.get('max_faces_num', 80000)
    save_memory = data.get('save_memory', False)
    do_texture_mapping = data.get('do_texture_mapping', False)
    do_render = data.get('do_render', False)

    assert not (text_prompt and image_prompt), "Text and image can only be given to one"
    assert text_prompt or image_prompt, "Text and image can only be given to one"

    os.makedirs(save_folder, exist_ok=True)

    # stage 1, text to image
    if text_prompt:
        res_rgb_pil = text_to_image_model(
            text_prompt,
            seed=t2i_seed,
            steps=t2i_steps
        )
        res_rgb_pil.save(os.path.join(save_folder, "img.jpg"))
    elif image_prompt:
        res_rgb_pil = Image.open(image_prompt)

    # stage 2, remove background
    res_rgba_pil = rembg_model(res_rgb_pil)
    res_rgba_pil.save(os.path.join(save_folder, "img_nobg.png"))

    # stage 3, image to views
    (views_grid_pil, cond_img), view_pil_list = image_to_views_model(
        res_rgba_pil,
        seed=gen_seed,
        steps=gen_steps
    )
    views_grid_pil.save(os.path.join(save_folder, "views.jpg"))

    # stage 4, views to mesh
    views_to_mesh_model(
        views_grid_pil,
        cond_img,
        seed=gen_seed,
        target_face_count=max_faces_num,
        save_folder=save_folder,
        do_texture_mapping=do_texture_mapping
    )

    # stage 5, render gif
    if do_render:
        gif_renderer(
            os.path.join(save_folder, 'mesh.obj'),
            gif_dst_path=os.path.join(save_folder, 'output.gif'),
        )

    return jsonify({"message": "3D generation completed", "save_folder": save_folder})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)

# 测试用的 curl 命令
# curl -X POST http://localhost:5000/generate3d -H "Content-Type: application/json" -d '{"text_prompt": "A beautiful landscape", "device": "cuda:1"}'
