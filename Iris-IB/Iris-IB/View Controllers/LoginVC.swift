//
//  LoginVC.swift
//  Iris-IB
//
//  Created by Ashwin Rathie on 11/5/19.
//  Copyright © 2019 Ashwin Rathie. All rights reserved.
//

import Foundation
import UIKit

class LoginVC : UIViewController {
    override func loadView() {
        // background


        var view = UILabel()

        view.frame = CGRect(x: 0, y: 0, width: 1496, height: 1671)

        view.backgroundColor = .white


        let image0 = UIImage(named: "image.png")?.cgImage

        let layer0 = CALayer()

        layer0.contents = image0

        layer0.bounds = view.bounds

        layer0.position = view.center

        view.layer.addSublayer(layer0)



        var parent = self.view!

        parent.addSubview(view)

        view.translatesAutoresizingMaskIntoConstraints = false

        view.widthAnchor.constraint(equalToConstant: 1496).isActive = true

        view.heightAnchor.constraint(equalToConstant: 1671).isActive = true

        view.leadingAnchor.constraint(equalTo: parent.leadingAnchor, constant: -550).isActive = true

        view.topAnchor.constraint(equalTo: parent.topAnchor, constant: -448).isActive = true
        
        
        



        
    }
}
