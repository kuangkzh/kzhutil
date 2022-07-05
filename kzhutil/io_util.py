from typing import AnyStr


def get_clipboard():
    """
    get clipboard text, works on win32

    :return: clipboard text
    """
    import win32clipboard
    import win32con
    win32clipboard.OpenClipboard()
    s = win32clipboard.GetClipboardData(win32con.CF_TEXT)
    win32clipboard.CloseClipboard()
    return s


def set_clipboard(s: AnyStr):
    """
    set clipboard text, works on win32

    :param s: clipboard text
    """
    import win32clipboard
    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardText(s)
    win32clipboard.CloseClipboard()
