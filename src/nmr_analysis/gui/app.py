from nicegui import ui
import os


def main():
    @ui.page("/")
    def index():
        ui.label("NMR Analysis Studio").classes("text-4xl font-bold text-center m-4")

        with ui.card().classes("w-full max-w-3xl mx-auto"):
            ui.label("Control Panel").classes("text-2xl")
            ui.separator()

            with ui.row().classes("w-full items-center"):
                ui.input("Data Directory", placeholder="Path/to/data").classes(
                    "flex-grow"
                )
                ui.button("Load", icon="folder_open")

            with ui.row().classes("w-full items-center"):
                ui.select(
                    ["T1 Inversion Recovery", "T2 Spin Echo", "T2* FID"],
                    label="Experiment Type",
                ).classes("flex-grow")
                ui.button("Analyze", icon="analytics")

        with ui.card().classes("w-full max-w-3xl mx-auto mt-4"):
            ui.label("Results").classes("text-xl")
            # Placeholder for plots
            ui.label("No results yet.").classes("text-gray-500")

    ui.run(title="NMR Analysis", port=8080, show=True)


if __name__ == "__main__":
    main()
