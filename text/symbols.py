""" from https://github.com/keithito/tacotron """

_pad = '_'
_punctuation = ';:,.!?¡¿—…"«»“” %-'  # ¡Guion añadido!
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
_letters_spanish = 'áéíóúüñÁÉÍÓÚÜÑ'
_numbers = '0123456789'

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa) + list(_letters_spanish) + list(_numbers)

# Special symbol ids
SPACE_ID = symbols.index(" ")