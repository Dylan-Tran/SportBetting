from Glicko2.Glicko2_parameters import INITIAL_RATING, TAO

from functools import lru_cache
from typing import Final, List
import math

GLICKO2_SCALING_FACTOR: Final = 173.7178


def convert_rating_to_glicko2_rating(rating):
    return (rating - INITIAL_RATING) / GLICKO2_SCALING_FACTOR


def convert_deviation_to_glicko2_deviation(deviation):
    return deviation / GLICKO2_SCALING_FACTOR


def convert_glicko2_rating_to_rating(glicko2_rating):
    return GLICKO2_SCALING_FACTOR * glicko2_rating + INITIAL_RATING


def convert_glicko2_deviation_to_deviation(glicko2_deviation):
    return GLICKO2_SCALING_FACTOR * glicko2_deviation


@lru_cache
def calc_g(deviation):
    return 1.0 / math.sqrt(1 + 3 * deviation**2 / math.pi**2)


@lru_cache
def calc_E(rating, opponent_rating, opponent_deviation):
    return 1.0 / (
        1 + math.exp(-calc_g(opponent_deviation) * (rating - opponent_rating))
    )


def calc_v(rating, opponent_ratings, opponent_deviations):
    def v_helper(opponent_rating, opponent_deviation):
        E = calc_E(rating, opponent_rating, opponent_deviation)
        return calc_g(opponent_deviation) ** 2 * E * (1 - E)

    return 1.0 / math.fsum(map(v_helper, opponent_ratings, opponent_deviations))


def calc_delta(rating, opponent_ratings, opponent_deviations, outcomes):
    def delta_helper(opponent_rating, opponent_deviation, outcome):
        return calc_g(opponent_deviation) * (
            outcome - calc_E(rating, opponent_rating, opponent_deviation)
        )

    v = calc_v(rating, opponent_ratings, opponent_deviations)
    return v * math.fsum(
        map(delta_helper, opponent_ratings, opponent_deviations, outcomes)
    )


def get_f(delta, deviation, v, volatility):
    return (
        lambda x: math.exp(x)
        * (delta**2 - deviation**2 - v - math.exp(x))
        / (2 * (deviation**2 + v + math.exp(x)) ** 2)
        - (x - math.log(volatility**2)) / TAO**2
    )


def calc_new_volatility(delta, deviation, v, volatility):
    f = get_f(delta, deviation, v, volatility)
    TOLERANCE = 0.000001

    A = math.log(volatility**2)
    B = None
    if delta**2 > deviation**2 + v:
        B = math.log(delta**2 - deviation**2 - v)
    else:
        k = 1
        while f(A - k * TAO) < 0:
            k += 1
        B = A - k * TAO

    f_a = f(A)
    f_b = f(B)
    while abs(B - A) > TOLERANCE:
        C = A + (A - B) * f_a / (f_b - f_a)
        f_c = f(C)

        if f_c * f_b <= 0:
            A = B
            f_a = f_b
        else:
            f_a = f_a / 2

        B = C
        f_b = f_c

    return math.exp(A / 2)


def calc_new_value_helper(deviation, new_volatility):
    return math.sqrt(deviation**2 + new_volatility**2)


def calc_new_deviation(v, deviation, new_volatility):
    return 1.0 / math.sqrt(
        1.0 / calc_new_value_helper(deviation, new_volatility) ** 2 + 1.0 / v
    )


def calc_new_rating(
    rating, new_deviation, opponent_ratings, opponent_deviations, outcomes
):
    def new_rating_helper(opponent_rating, opponent_deviation, outcome):
        return calc_g(opponent_deviation) * (
            outcome - calc_E(rating, opponent_rating, opponent_deviation)
        )

    return rating + new_deviation**2 * math.fsum(
        map(new_rating_helper, opponent_ratings, opponent_deviations, outcomes)
    )


"""
if player does not compete during rating period then use this function
"""


def no_compete_new_rating(old_rating) -> float:
    return old_rating


def no_compete_new_deviation(old_deviation, old_volatility) -> float:
    return math.sqrt(old_deviation**2 + old_volatility**2)


def no_compete_new_volatility(old_volatility) -> float:
    return old_volatility


def glicko2_evaluate_no_complete(rating, deviation, volatility) -> List[float]:
    return [
        no_compete_new_rating(rating),
        no_compete_new_deviation(deviation, volatility),
        no_compete_new_volatility(volatility),
    ]


def glicko2_evaluate(
    rating, deviation, volatility, opponent_ratings, opponent_deviations, outcomes
) -> List[float]:
    glicko2_rating = convert_rating_to_glicko2_rating(rating)
    glicko2_deviation = convert_deviation_to_glicko2_deviation(deviation)
    opponent_glicko2_ratings = list(
        map(convert_rating_to_glicko2_rating, opponent_ratings)
    )
    opponent_glicko2_deviations = list(
        map(convert_deviation_to_glicko2_deviation, opponent_deviations)
    )

    v = calc_v(glicko2_rating, opponent_glicko2_ratings, opponent_glicko2_deviations)
    delta = calc_delta(
        glicko2_rating, opponent_glicko2_ratings, opponent_glicko2_deviations, outcomes
    )

    new_volatility = calc_new_volatility(delta, glicko2_deviation, v, volatility)
    new_glicko2_deviation = calc_new_deviation(v, glicko2_deviation, new_volatility)
    new_glicko2_rating = calc_new_rating(
        glicko2_rating,
        new_glicko2_deviation,
        opponent_glicko2_ratings,
        opponent_glicko2_deviations,
        outcomes,
    )

    new_rating = convert_glicko2_rating_to_rating(new_glicko2_rating)
    new_deviation = convert_glicko2_deviation_to_deviation(new_glicko2_deviation)

    return new_rating, new_deviation, new_volatility


def main(
    rating, deviation, volatility, opponent_ratings, opponent_deviations, outcomes
):
    rating = convert_rating_to_glicko2_rating(rating)
    deviation = convert_deviation_to_glicko2_deviation(deviation)
    opponent_ratings = list(map(convert_rating_to_glicko2_rating, opponent_ratings))
    opponent_deviations = list(
        map(convert_deviation_to_glicko2_deviation, opponent_deviations)
    )

    v = calc_v(rating, opponent_ratings, opponent_deviations)
    delta = calc_delta(rating, opponent_ratings, opponent_deviations, outcomes)

    new_volatility = calc_new_volatility(delta, deviation, v, volatility)
    new_deviation = calc_new_deviation(v, deviation, new_volatility)
    new_rating = calc_new_rating(
        rating, new_deviation, opponent_ratings, opponent_deviations, outcomes
    )

    new_rating = convert_glicko2_rating_to_rating(new_rating)
    new_deviation = convert_glicko2_deviation_to_deviation(new_deviation)
    return new_rating, new_deviation, new_volatility
