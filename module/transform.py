from PIL import Image

def resize_with_padding(img, target_size):
    """
    이미지에 여백을 추가하여 원하는 크기로 조정합니다.
    """
    original_width, original_height = img.size 

    target_width, target_height = target_size

    ratio_img = original_width / original_height
    ratio_target = target_width / target_height

    if ratio_img > ratio_target:
        resize_width = target_width
        resize_height = round(resize_width / ratio_img)
    else:
        resize_height = target_height
        resize_width = round(resize_height * ratio_img)

    img_resized = img.resize((resize_width, resize_height), Image.Resampling.LANCZOS)

    background = Image.new("RGB", target_size, (255, 255, 255))
    offset = ((target_width - resize_width) // 2, (target_height - resize_height) // 2)
    background.paste(img_resized, offset)

    return background
