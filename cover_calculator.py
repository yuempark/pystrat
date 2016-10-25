import math
import pandas as pd
import numpy as np
from calculation_functions import vincenty_inverse
from calculation_functions import compass_bearing
from calculation_functions import fisher_mean

def cover_calculator(covers_csv_string):
    """
    Given the following for the start and end points, calculate a stratigraphic thickness between the two points:
    - latitude (decimal degrees)
    - longitude (decimal degrees)
    - elevation (m)
    - strike of bedding (RHR)
    - dip of bedding

    Input .csv must follow the format of the given template, 'covers_template.csv'.

    Example to run from the command line:
    python -c 'import cover_calculator; cover_calculator.cover_calculator("Users/yuempark/Documents/Hongzixi_covers.csv")'
    """

    # read in the data
    data = pd.read_csv(covers_csv_string)

    for i in range(len(data.index)):

        # vincenty distance, converted from km to m
        data.loc[i,'s'] = vincenty_inverse((data['start_lat'][i],data['start_lon'][i]),\
                                           (data['end_lat'][i],data['end_lon'][i]))
        data.loc[i,'s'] = data.loc[i,'s'] * 1000

        # compass bearing
        data.loc[i,'bearing'] = compass_bearing((data['start_lat'][i],data['start_lon'][i]),\
                                                (data['end_lat'][i],data['end_lon'][i]))

        # first convert strike to dip direction
        data.loc[i,'start_dip_direction'] = (data['start_strike'][i] + 90) % 360
        data.loc[i,'end_dip_direction'] = (data['end_strike'][i] + 90) % 360

        # convert dip direction and dip of bedding to a pole (perpendicular to bedding) trend and plunge
        data.loc[i,'start_pole_trend'] = (data['start_dip_direction'][i] + 180) % 360
        data.loc[i,'start_pole_plunge'] = 90 - data['start_dip'][i]
        data.loc[i,'end_pole_trend'] = (data['end_dip_direction'][i] + 180) % 360
        data.loc[i,'end_pole_plunge'] = 90 - data['end_dip'][i]

        # get the mean pole
        di_block = [[data['start_pole_trend'][i],data['start_pole_plunge'][i]],\
                    [data['end_pole_trend'][i],data['end_pole_plunge'][i]]]
        fpars = fisher_mean(di_block)
        data.loc[i,'mean_pole_trend'] = fpars['dec']
        data.loc[i,'mean_pole_plunge'] = fpars['inc']

        # convert back to mean dip direction and dip
        data.loc[i,'mean_dip_direction'] = (data['mean_pole_trend'][i] + 180) % 360
        data.loc[i,'mean_dip'] = 90 - data['mean_pole_plunge'][i]

        # distance perpendicular to strike
        data.loc[i,'perp_s'] = data['s'][i] * math.cos(math.radians(abs(data['mean_dip_direction'][i] -\
                                                                        data['bearing'][i])))

        # elevation difference
        data.loc[i,'d_elev'] = data['end_elev'][i] - data['start_elev'][i]

        # true perpendicular distance (accounting for elevation)
        data.loc[i,'R'] = math.sqrt(data['d_elev'][i]**2 + data['perp_s'][i]**2)

        # absolute inclination between the adjusted points
        data.loc[i,'inclination'] = math.degrees(math.atan(data['d_elev'][i] / data['perp_s'][i]))

        # the inclination from the line between the adjusted points
        if data['d_elev'][i] > 0:
            data.loc[i,'angle'] = 90 - data['inclination'][i] - data['mean_dip'][i]
        else:
            data.loc[i,'angle'] = 90 - data['mean_dip'][i] + data['inclination'][i]

        # the stratigraphic height
        data.loc[i,'HEIGHT'] = round(data['R'][i] * math.cos(math.radians(data['angle'][i])), 1)

    # save the .csv
    data.to_csv(covers_csv_string + '_calculated.csv', index=False)
