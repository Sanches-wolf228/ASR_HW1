import editdistance

def calc_cer(target_text, predicted_text) -> float:
    splitted_target = target_text.split(' ')
    if len(splitted_target) == 0:
        return 1
    return editdistance.distance(splitted_target, predicted_text.split(' ')) / len(splitted_target)

def calc_wer(target_text, predicted_text) -> float:
    if len(target_text) == 0:
        return 1
    return editdistance.distance(target_text, predicted_text) / len(target_text)
