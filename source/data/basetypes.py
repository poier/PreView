"""
Predifined datatypes

Copyright 2015, 2018 ICG, Graz University of Technology

This file is part of PreView.

PreView is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

PreView is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PreView.  If not, see <http://www.gnu.org/licenses/>.
"""

from collections import namedtuple


ICVLFrame = namedtuple('ICVLFrame',
                       ['dpt','gtorig','gtcrop','T','gt3Dorig',
                       'gt3Dcrop','com','fileName','subSeqName','config'])

ICGFrame = namedtuple('ICGFrame',
					['img_depth', 'gt_uvd', 'gt_uvd_crop', 'T', 
		                 'gt_3D', 'gt_3D_crop', 'com_3D', 'filename',
		                 'config', 'handtype', 'id_camparams'])
                   
NamedImgSequence = namedtuple('NamedImgSequence',['name','data','config'])
        
class Arguments(object):
    pass
