from googletrans import Translator


# Translate text
def sinhala_translate(text, dest='si'):
    translator = Translator()
    return translator.translate(text, dest=dest).text


def spanish_translate(text, dest='es'):
    translator = Translator()
    return translator.translate(text, dest=dest).text


def combined_translation(text):
    sinhala = sinhala_translate(text)
    spanish = spanish_translate(text)

    # add two string value characters randomly

    encrypted_text = sinhala[:3] + spanish[:3] + sinhala[3:] + spanish[3:] + sinhala[6:] + spanish[6:]

    return encrypted_text


# Main
if __name__ == '__main__':
    text = 'Hello World'
    print(combined_translation(text))
    print(text)
