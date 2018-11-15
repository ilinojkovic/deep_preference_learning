import time
import xml.etree.ElementTree as ET
import pandas as pd
import geocoder
from optparse import OptionParser


class XML2DataFrame:
    def __init__(self, xml_path):
        xml = ET.parse(xml_path)
        self.root = xml.getroot()[0]
        self.counter = 0

    def parse_root(self, root):
        """Return a list of dictionaries from the text
         and attributes of the children under this XML root."""
        parsed_list = []
        for i, child in enumerate(root):
            self.index = i
            parsed_list.append(self.parse_element(child))
            if (self.index + 1) % 500 == 0:
                print(self.index)
        return parsed_list

    def parse_element(self, element, parsed=None, parent_tag=None):
        """ Collect {key:attribute} and {tag:text} from the XML
         element and all its children into a single dictionary of strings."""
        if parsed is None:
            parsed = dict()

        if element.tag == 'attachments':
            parsed[element.tag] = len(element)
            return parsed

        for key in element.keys():
            if key not in parsed:
                if parent_tag is not None:
                    parsed_key = '{}-{}'.format(parent_tag, key)
                else:
                    parsed_key = key
                parsed[parsed_key] = element.attrib.get(key)
            else:
                raise ValueError('duplicate attribute \'{0}\':\'{1}\' at element {2}\nCurrently parsed {3}'.format(key,
                                                                                                                   element.attrib.get(
                                                                                                                       key),
                                                                                                                   element.tag,
                                                                                                                   parsed))

        """ Apply recursion """
        for child in list(element):
            # Parse address
            if child.tag == 'address':
                address = []
                address_info = ['streetNumber', 'street', 'postalCode', 'locality', 'region', 'country']
                for info in address_info:
                    info_element = child.find(info)
                    if info_element is not None and info_element.text is not None:
                        address.append(info_element.text)
                address_str = ' '.join(address)
                parsed['address'] = address_str
                # geo_address = geocoder.arcgis(address_str)
                # if geo_address:
                #     parsed['lat'] = geo_address.latlng[0]
                #     parsed['lng'] = geo_address.latlng[1]
                # else:
                #     # Try again after sleep
                #     time.sleep(5)
                #     geo_address = geocoder.arcgis(address_str)
                #     if geo_address:
                #         parsed['lat'] = geo_address.latlng[0]
                #         parsed['lng'] = geo_address.latlng[1]
                #     else:
                #         self.counter += 1
                #         print('Address not found:', self.index, self.counter, len(child), child.tag, address,
                #               address_str)
            elif len(child) == 0:
                if child.tag == 'type':
                    if child.text != 'buy' and child.text != 'rent':
                        raise ValueError('Undefined type: {}'.format(child.text))
                    parsed[child.tag] = 0 if child.text == 'buy' else 1
                elif child.tag in ['feature', 'category', 'utility']:
                    if child.text not in parsed:
                        parsed['{}-{}'.format(child.tag, child.text)] = 1
                    else:
                        raise ValueError('duplicate feature {0}. Currently parsed {1}'.format(child.text, parsed))
                elif child.tag == 'value':
                    if child.attrib['key'] not in parsed:
                        parsed['{}-{}'.format(child.tag, child.attrib['key'])] = child.text
                    else:
                        raise ValueError(
                            'duplicate value {0}. Currently parsed {1}'.format(child.attrib['key'], parsed))
                elif child.tag not in parsed:
                    if parent_tag is not None:
                        parsed_key = '{}-{}'.format(parent_tag, child.tag)
                    else:
                        parsed_key = child.tag
                    parsed[parsed_key] = child.text
                else:
                    raise ValueError(
                        'duplicate tag \'{0}\':\'{1}\' at element {2}\nCurrently parsed {3}'.format(child.tag,
                                                                                                    child.text,
                                                                                                    element.tag,
                                                                                                    parsed))
            else:
                if parent_tag is not None:
                    parsed_key = '{}-{}'.format(parent_tag, child.tag)
                else:
                    parsed_key = child.tag
                self.parse_element(child, parsed, parsed_key)

        return parsed

    def process_data(self):
        """ Initiate the root XML, parse it, and return a dataframe"""
        structure_data = self.parse_root(self.root)
        return pd.DataFrame(structure_data)


def create_parser():
    parser = OptionParser()
    parser.add_option('-i', '--input',
                      dest='input',
                      default='../data/raw/swissrets1.xml',
                      help='Path to .xml file to extract property data from.')
    parser.add_option('-o', '--output',
                      dest='output',
                      default='../data/processed/extracted_data.pkl',
                      help='Path to output .pkl file with extracted pandas dataframe.')
    return parser


def main():
    # Load command input options
    options, remainder = create_parser().parse_args()

    # Extract data
    xml2df = XML2DataFrame(options.input)
    xml_dataframe = xml2df.process_data()

    # Save pandas dataframe
    xml_dataframe.to_pickle(options.output)


if __name__ == '__main__':
    main()
