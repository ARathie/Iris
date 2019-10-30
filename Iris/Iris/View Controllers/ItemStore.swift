//
//  ItemStore.swift
//  Iris
//
//  Created by Ashwin Rathie on 10/30/19.
//  Copyright © 2019 Ashwin Rathie. All rights reserved.
//

import Foundation
import UIKit

class ItemStore {
    var allItems = [Item] ()
    
    func removeItem(_ item: Item) {
        if let index = allItems.index(of: item) {
            allItems.remove(at: index)
        }
    }
    
    func moveItem(from fromIndex: Int, to toIndex: Int) {
        if fromIndex == toIndex {
            return
        }
        // Get reference to object being moved so you can reinsert it
        let movedItem = allItems[fromIndex]
        // Remove item from array
        allItems.remove(at: fromIndex)
        // Insert item in array at new location
        allItems.insert(movedItem, at: toIndex)
    }
    
    //@discardableResult annotation means that a caller of this function is free to ignore the result of calling this functio
    @discardableResult func createItem() -> Item {
        let newItem = Item(random: true)
        allItems.append(newItem)
        return newItem
    }
}
