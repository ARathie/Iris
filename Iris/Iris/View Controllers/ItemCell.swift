//
//  ItemCell.swift
//  Iris
//
//  Created by Ashwin Rathie on 10/30/19.
//  Copyright © 2019 Ashwin Rathie. All rights reserved.
//

import Foundation
import UIKit
class ItemCell: UITableViewCell {
    
    @IBOutlet var nameLabel: UILabel!
    @IBOutlet var serialNumberLabel: UILabel!
    @IBOutlet var valueLabel: UILabel!
    
    override func awakeFromNib() {
        super.awakeFromNib()
        nameLabel.adjustsFontForContentSizeCategory = true
        serialNumberLabel.adjustsFontForContentSizeCategory = true
        valueLabel.adjustsFontForContentSizeCategory = true
    }
}
