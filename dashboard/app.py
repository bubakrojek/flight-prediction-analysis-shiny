import faicons as fa
import polars as pl
from faicons import icon_svg
from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

flights = pl.scan_parquet(os.path.join(BASE_DIR, "data", "flights.parquet"))
airlines = pl.scan_parquet(os.path.join(BASE_DIR, "data", "airlines.parquet"))
airports = pl.scan_parquet(os.path.join(BASE_DIR, "data", "airports.parquet"))

from utils.data_preprocessing import get_airports_list, get_airlines_list, filter_airlines_with_avg_delay, \
    average_airlines_delay_plot, filter_airlines_with_delay, airlines_delay_distribution_plot, \
    filter_delays_per_seasons, delay_and_cancel_seasonality_plot, delays_or_cancellation_w_month_and_dow, heatmap_plot, \
    get_states_list, filter_airports, outgoing_flights_plot, group_airport_pairs, trails_plot, \
    filter_top_10_pairs_with_avg_delay, average_pairs_delay_plot, get_flight_info, flight_visualisation, \
    amount_delay_types_plot, get_airport_delay_median, get_airline_delay_median
from utils.data_preprocessing import filter_airport_with_delay, filter_airports_with_divided_delay
from utils.data_preprocessing import airport_delay_distribution_plot, airport_delays_divided_by_occurrence_plot
from utils.data_preprocessing import get_dataset_summary

airport_choices = get_airports_list(airports)
airline_choices = get_airlines_list(airlines)
state_choices = get_states_list(airports)

app_ui = ui.page_fluid(
    ui.h2("Flights analysis dashboard"),
    ui.navset_tab(
        ui.nav_panel('Dataset summary',
                     ui.layout_column_wrap(

                         ui.card(
                             ui.card_header("Number of flights"),
                             ui.div(
                                 ui.output_text("number_of_flights"),
                                 style="""
                                     display: flex; 
                                     justify-content: center; 
                                     align-items: center; 
                                     height: 100%; 
                                     font-size: 3rem; 
                                     font-weight: bold; 
                                     color: #007bc0;
                                 """
                             ),
                             full_screen=True
                         ),

                         ui.card(
                             ui.card_header("Glimpse at data"),
                             ui.output_data_frame("glimpse_at_data"),
                             full_screen=True,
                             style="height: 100%;"
                         ),

                         ui.card(
                             ui.card_header("Average delay"),
                             ui.div(
                                 ui.output_text("average_delay"),
                                 style="""
                                     display: flex; 
                                     justify-content: center; 
                                     align-items: center; 
                                     height: 100%; 
                                     font-size: 3rem; 
                                     font-weight: bold; 
                                     color: #e45756;
                                 """
                             ),
                             full_screen=True
                         ),

                         ui.card(
                             ui.card_header("Percentage of cancelled"),
                             ui.div(
                                 ui.output_text("percentage_of_cancelled"),
                                 style="""
                                     display: flex; 
                                     justify-content: center; 
                                     align-items: center; 
                                     height: 100%; 
                                     font-size: 3rem; 
                                     font-weight: bold; 
                                     color: #f2a900;
                                 """
                             ),
                             full_screen=True
                         ),

                         width=1 / 2,
                         fill=True,
                         heights_equal="all",
                         style="height: calc(100vh - 120px); padding: 15px;"

                     )
                     ),
        ui.nav_panel('Airports',
                     ui.h3('Airports'),
                     ui.layout_sidebar(
                         ui.sidebar(
                             ui.input_select(
                                 id='airport',
                                 label="Choose airport",
                                 choices=airport_choices,
                                 selected=airport_choices[0]
                             ),
                             ui.hr(),
                             ui.input_select(
                                 id='states',
                                 label="Choose states",
                                 choices=state_choices,
                                 selected=state_choices,
                                 multiple=True
                             ),
                             ui.hr(),
                             ui.input_numeric(
                                 id='flight_amount',
                                 label='Enter minimum outgoing flight number',
                                 value=0,
                             ),
                         ),
                         output_widget('render_airports_map'),
                         ui.hr(),
                         ui.h3("Data related to chosen airport"),
                         ui.layout_columns(
                             ui.card(
                                 ui.output_plot('airport_delay_violin', height="100%"),
                                 full_screen=True

                             ),
                             ui.card(
                                 ui.layout_column_wrap(
                                     ui.value_box(
                                         ui.span("Delays Median", style="font-size: 1.2rem; font-weight: 500;"),
                                         ui.div(
                                             ui.output_text('delays_median'),
                                             style="font-size: 4rem; font-weight: bold; line-height: 1.2;"
                                         ),
                                         showcase=icon_svg("clock"),
                                         theme="bg-primary",
                                         showcase_layout="bottom"),
                                     output_widget('render_amount_delay_types_plot', ),
                                     fill=True,
                                     fillable=True,
                                     width=1
                                 )
                             ),
                             col_widths=(6, 6)
                         ),
                         output_widget('airport_delay_types')

                     )
                     ),

        ui.nav_panel('Lines',
                     ui.h3('Lines delays'),
                     ui.layout_sidebar(
                         ui.sidebar(
                             ui.input_slider(
                                 id="delay_range",
                                 label="Arrival delay range [min]",
                                 min=-10,
                                 max=20,
                                 value=[-10, 20],
                                 step=1,
                             ),
                             ui.hr(),
                             ui.input_select(
                                 id='airlines_multi',
                                 label="Choose multiple airlines",
                                 choices=airline_choices,
                                 selected=airline_choices[:3],
                                 multiple=True
                             ),
                             ui.hr(),
                             ui.input_select(
                                 id='airline',
                                 label="Choose single airline",
                                 choices=airline_choices,
                                 selected=airline_choices[0]
                             ),

                         ),

                         ui.output_plot('airlines_delay_barplot'),
                         ui.hr(),
                         ui.layout_columns(
                             ui.card(
                                 ui.output_plot('airline_delay_violin', height="800px")),
                             ui.card(
                                 ui.value_box(
                                     ui.div(
                                         "Delays Median",
                                         style="font-size: clamp(1rem, 1.5vw, 1.5rem); font-weight: 500; text-align: top; opacity: 0.9;"
                                     ),

                                     ui.div(
                                         ui.output_text('airline_delays_median'),
                                         style="""
                                                 font-size: clamp(4rem, 8vw, 12rem); 
                                                 font-weight: 900; 
                                                 line-height: 1; 
                                                 text-align: center;
                                                 display: flex;
                                                 align-items: center;
                                                 justify-content: center;
                                                 flex-grow: 1; /* Pozwala wartości zająć całą resztę pustego miejsca */
                                             """
                                     ),

                                     showcase=icon_svg("clock"),
                                     theme="bg-primary",
                                     showcase_layout="bottom",

                                     class_="text-center d-flex flex-column"
                                 )
                             )
                         )
                     )
                     ),
        ui.nav_panel('Seasonality',
                     ui.layout_sidebar(
                         ui.sidebar(
                             ui.input_slider(
                                 id="months_range",
                                 label="Choose month range",
                                 min=1,
                                 max=12,
                                 value=[1, 12],
                                 step=1,
                             ),
                             ui.hr(),
                             ui.input_select(
                                 id='event_type',
                                 label="Choose type of event",
                                 choices=['Delay', 'Cancellation'],
                                 selected='Delay'
                             )

                         ),
                         output_widget('render_radial_month'),
                         ui.row(
                             ui.column(9, output_widget('render_heatmap_dow_month')),
                             ui.column(3,
                                       ui.layout_column_wrap(
                                           ui.value_box(
                                               'Max value',
                                               ui.output_text('max_value'),
                                               fa.icon_svg("arrow-up"),
                                               theme="failure"
                                           ),
                                           ui.value_box(
                                               'Mean value',
                                               ui.output_text('mean_value'),
                                               fa.icon_svg("percent"),
                                               theme="primary"
                                           ),
                                           ui.value_box(
                                               'Min value',
                                               ui.output_text('min_value'),
                                               fa.icon_svg("arrow-down"),
                                               theme="success"
                                           ),
                                           width=1,
                                           fill=True,
                                           heights_equal="all",
                                           style="height: calc(100vh - 120px); padding: 15px;"
                                       )
                                       )
                         )

                     )
                     ),

        ui.nav_panel('Routes',
                     ui.layout_sidebar(
                         ui.sidebar(
                             ui.input_select(
                                 id='chart_mode',
                                 label="Choose mode of chart",
                                 choices=['Top', 'Certain Airport'],
                                 selected='Certain Airport'
                             ),
                             ui.input_numeric(
                                 id='top_airports',
                                 label='Enter number of pairs',
                                 value=10,
                             ),
                             ui.input_slider(
                                 id="pairs_departure_delay_range",
                                 label="Choose departure delay range",
                                 min=-60,
                                 max=60,
                                 value=[-60, 60],
                                 step=1,
                             ),

                             ui.input_slider(
                                 id="pairs_arrival_delay_range",
                                 label="Choose arrival delay range",
                                 min=-60,
                                 max=60,
                                 value=[-60, 60],
                                 step=1,
                             ),
                             ui.hr(),
                             ui.input_select(
                                 id='pairs_airport',
                                 label="Choose single airport",
                                 choices=airport_choices,
                                 selected=airport_choices[0]
                             ),

                         ),
                         ui.layout_columns(

                             ui.card(
                                 ui.card_header("Flight connections map", class_="bg-light"),
                                 ui.markdown(
                                     "The map to the right displays the flight network for the selected airport. \n\n"
                                     "**Line thickness** indicates the traffic volume (total number of flights in both directions). \n\n"
                                     "Hover over a line or point to see the exact airport names and the specific number of flights."
                                 )
                             ),
                             ui.card(
                                 ui.div(
                                     output_widget('render_pairs', width="100%", height="100%"),

                                     style="flex-grow: 1; display: flex; flex-direction: column;"
                                 ),
                                 full_screen=True,

                                 class_="p-0 d-flex flex-column",
                                 min_height="600px"
                             ),
                             ui.card(
                                 ui.card_header("Route delays (Departures vs Arrivals)", class_="bg-light"),
                                 ui.markdown(
                                     "This bar chart compares the **median delay** for specific airport pairs. "
                                     "Note that **negative values (below zero)** indicate that flights on a given route "
                                     "typically depart or arrive *ahead of schedule*."
                                 ),
                                 output_widget('render_pairs_delay_barplot'),
                                 full_screen=True
                             ),
                             col_widths=(4, 8, 12)
                         )

                     )
                     ),

        ui.nav_panel('Flight summary',
                     ui.layout_sidebar(
                         ui.sidebar(
                             ui.input_text(
                                 id='flight_number',
                                 label="Input flight number",
                                 value="98",
                                 autocomplete="on"
                             ),
                             ui.hr(),
                             ui.input_select(
                                 id='day',
                                 label="Choose day of flight",
                                 choices=[str(i) for i in range(1, 31)],
                                 selected='1'
                             ),
                             ui.hr(),
                             ui.input_select(
                                 id='month',
                                 label="Choose month of flight",
                                 choices=[str(i) for i in range(1, 13)],
                                 selected='1'
                             ),
                             ui.hr(),
                             ui.input_select(
                                 id='year',
                                 label="Choose year of flight",
                                 choices=["2015"],
                                 selected='2015'
                             ),
                             ui.hr(),
                             ui.input_select(
                                 id='airline_flight',
                                 label="Choose airline",
                                 choices=airline_choices,
                                 selected='AS'
                             ),

                         ),
                         output_widget('render_flight_visualisation')

                     )
                     ),

    ),

)


def server(input, output, session):
    # ---------------------------------------------------------------------------------------------
    # Podsumowanie zbioru
    @reactive.calc
    def data_summary():
        return get_dataset_summary(flights)

    @render.text
    def number_of_flights():
        return f"{data_summary()['number_of_flights']}"

    @render.data_frame
    def glimpse_at_data():
        return data_summary()['glimpse_at_data']

    @render.text
    def average_delay():
        return f"{data_summary()['average_delay']}"

    @render.text
    def percentage_of_cancelled():
        return f"{data_summary()['percentage_of_cancelled']}%"

    # ---------------------------------------------------------------------------------------------
    # Lotniska
    @reactive.calc
    def airport_data():
        selected = input.airport()

        return filter_airport_with_delay(flights, airports, selected)

    @output
    @render.plot
    def airport_delay_violin():
        data = airport_data()

        return airport_delay_distribution_plot(data)

    @reactive.calc
    def airports_flight_number_data():
        amount = input.flight_amount()
        states = input.states()
        if len(states) == len(state_choices):
            states = 'Wszystkie'
        return filter_airports(flights, airports, amount, states)

    @render_widget
    def render_airports_map():
        data = airports_flight_number_data()
        return outgoing_flights_plot(data)

    @render.text
    def delays_median():
        data = airport_data()

        return f"{get_airport_delay_median(data)['ARRIVAL_DELAY'][0]}"

    @reactive.calc
    def airport_delays_data():
        selected = input.airport()
        return filter_airports_with_divided_delay(flights, airports, selected)

    @render_widget
    def render_amount_delay_types_plot():
        data = airport_delays_data()

        return amount_delay_types_plot(data)

    @render_widget
    def airport_delay_types():
        data = airport_delays_data()
        return airport_delays_divided_by_occurrence_plot(data)

    # ---------------------------------------------------------------------------------------------
    # Linie
    @reactive.calc
    def airlines_delays():
        delay_range = input.delay_range()
        airline = input.airlines_multi()
        if len(airline) == len(airline_choices):
            airline = 'Wszystkie'

        return filter_airlines_with_avg_delay(
            flights,
            airlines,
            min_delay=delay_range[0],
            max_delay=delay_range[1],
            airlines_list=airline
        )

    @output
    @render.plot
    def airlines_delay_barplot():
        data = airlines_delays()
        return average_airlines_delay_plot(data)

    @reactive.calc
    def airline_delay():
        airline = input.airline()

        return filter_airlines_with_delay(
            flights,
            airlines,
            airline,
        )

    @output
    @render.plot
    def airline_delay_violin():
        data = airline_delay()
        return airlines_delay_distribution_plot(data)

    @render.text
    def airline_delays_median():
        data = airline_delay()

        return f"{get_airline_delay_median(data)['ARRIVAL_DELAY'][0]}"

    # ---------------------------------------------------------------------------------------------
    # Sezonowość

    @reactive.calc
    def month_data():
        months_range = input.months_range()
        event_type = input.event_type()

        return filter_delays_per_seasons(flights, months_range[0], months_range[1], event_type)

    @render_widget
    def render_radial_month():
        data = month_data()
        return delay_and_cancel_seasonality_plot(data)

    @reactive.calc
    def month_dow_data():
        event_type = input.event_type()
        return delays_or_cancellation_w_month_and_dow(flights, event_type)

    @render_widget
    def render_heatmap_dow_month():
        data = month_dow_data()
        return heatmap_plot(data['dataframe'])

    @render.text
    def max_value():
        return f"{round(month_dow_data()['max_value']['MEAN'][0], 2)}"

    @render.text
    def mean_value():
        return f"{round(month_dow_data()['mean_value']['MEAN'][0], 2)}"

    @render.text
    def min_value():
        return f"{round(month_dow_data()['min_value']['MEAN'][0], 2)}"

    # ---------------------------------------------------------------------------------------------
    # Trasy

    # Wizualizacja par
    @reactive.calc
    def route_pairs_data():
        chart_mode = input.chart_mode()
        top_airports = input.top_airports()
        pairs_airport = input.pairs_airport()

        return group_airport_pairs(flights, airports, chart_mode, top_airports, pairs_airport)

    @render_widget
    def render_pairs():
        data = route_pairs_data()
        return trails_plot(data)

    # Spoznienia par
    @reactive.calc
    def pairs_delay_data():
        min_departure_delay = input.pairs_departure_delay_range()[0]
        max_departure_delay = input.pairs_departure_delay_range()[1]
        min_arrival_delay = input.pairs_arrival_delay_range()[0]
        max_arrival_delay = input.pairs_arrival_delay_range()[1]
        pairs_airport = input.pairs_airport()

        return filter_top_10_pairs_with_avg_delay(flights,
                                                  airports,
                                                  min_departure_delay,
                                                  max_departure_delay,
                                                  min_arrival_delay,
                                                  max_arrival_delay,
                                                  pairs_airport
                                                  )

    @render_widget
    def render_pairs_delay_barplot():
        data = pairs_delay_data()

        return average_pairs_delay_plot(data)

    # ---------------------------------------------------------------------------------------------
    # Wizualizacja lotu

    @reactive.calc
    def flight_data():
        flight_number = input.flight_number()
        day = input.day()
        month = input.month()
        year = input.year()
        airline = input.airline_flight()

        return get_flight_info(flights, airports, flight_number, day, month, year, airline)

    @render_widget
    def render_flight_visualisation():
        data = flight_data()
        return flight_visualisation(data)


app = App(app_ui, server)
