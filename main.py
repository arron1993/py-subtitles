import flet as ft

from video import Video


def main(page: ft.Page):
    def on_pick_file(event):
        path = event.files[0].path
        selected_files.value = path
        selected_files.update()
        v = Video(path)
        audio_trans = v.get_audio_text()

    pick_files_dialog = ft.FilePicker(on_result=on_pick_file)
    selected_files = ft.Text()

    page.overlay.append(pick_files_dialog)

    page.add(
        ft.Row(
            [
                ft.ElevatedButton(
                    "Pick file",
                    icon=ft.icons.UPLOAD_FILE,
                    on_click=lambda _: pick_files_dialog.pick_files(),
                ),
                selected_files,
            ]
        )
    )


ft.app(main)
