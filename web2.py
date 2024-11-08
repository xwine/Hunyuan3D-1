# 开源模型根据Apache许可证版本2.0和其中第三方组件的其他许可证授权：
# 本分发中的以下模型可能已被THL A29 Limited（"腾讯修改"）修改。所有腾讯修改版权归THL A29 Limited 2024年所有。

# 版权归2024年THL A29 Limited，腾讯公司所有。保留所有权利。
# 本分发中的以下软件和/或模型可能已被THL A29 Limited（"腾讯修改"）修改。
# 所有腾讯修改版权归THL A29 Limited所有。

# Hunyuan 3D根据TENCENT HUNYUAN非商业许可证协议授权，
# 除了下面列出的第三方组件。
# Hunyuan 3D不对这些第三方组件的各自许可证中概述的内容施加任何额外限制。
# 用户必须遵守这些第三方组件原始许可证的所有条款和条件，
# 并且必须确保第三方组件的使用符合所有相关法律法规。

# 为避免疑问，Hunyuan 3D是指大型语言模型及其软件和算法，
# 包括训练模型权重、参数（包括优化器状态）、机器学习模型代码、推理启用代码、训练启用代码、
# 微调启用代码以及腾讯根据TENCENT HUNYUAN社区许可证协议公开提供的上述内容的其他元素。

import os
from PIL import Image
from flask import Flask, request, jsonify, render_template_string, send_from_directory, abort
from infer import Text2Image, Removebg, Image2Views, Views2Mesh, GifRenderer

app = Flask(__name__)

# 初始化模型
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
    save_folder = data.get('save_folder', './static/')
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

    # 阶段1，文字转图像
    if text_prompt:
        res_rgb_pil = text_to_image_model(
            text_prompt,
            seed=t2i_seed,
            steps=t2i_steps
        )
        res_rgb_pil.save(os.path.join(save_folder, "img.jpg"))
    elif image_prompt:
        res_rgb_pil = Image.open(image_prompt)

    # 阶段2，去除背景
    res_rgba_pil = rembg_model(res_rgb_pil)
    res_rgba_pil.save(os.path.join(save_folder, "img_nobg.png"))

    # 阶段3，图像转视图
    (views_grid_pil, cond_img), view_pil_list = image_to_views_model(
        res_rgba_pil,
        seed=gen_seed,
        steps=gen_steps
    )
    views_grid_pil.save(os.path.join(save_folder, "views.jpg"))

    # 阶段4，视图转网格
    views_to_mesh_model(
        views_grid_pil,
        cond_img,
        seed=gen_seed,
        target_face_count=max_faces_num,
        save_folder=save_folder,
        do_texture_mapping=do_texture_mapping
    )

    # 阶段5，渲染gif
    if do_render:
        gif_renderer(
            os.path.join(save_folder, 'mesh.obj'),
            gif_dst_path=os.path.join(save_folder, 'output.gif'),
        )

    return jsonify({"message": "3D generation completed", "url": f"/view3d/mesh.obj"})


@app.route('/view3d/<filename>')
def view_3d(filename):
    obj_url = f"static/{filename}"
    # 从 URL 参数获取文件路径（默认为 static/model.obj）
    return render_template('index.html', file_path=obj_url)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)

# curl 示例
# 生成3D图像:
# curl -X POST http://localhost:8080/generate3d -H "Content-Type: application/json" -d '{"text_prompt": "a cute dog"}'
# 查看3D图像:
# 在浏览器中访问 http://localhost:8080/view3d/mesh.obj
